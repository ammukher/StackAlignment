
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <assert.h>
#include <stdlib.h>
#include <iomanip>

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkCenteredTransformInitializer.h"
#include "itkMedianImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkCenteredRigid2DTransform.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkNearestNeighborExtrapolateImageFunction.h"
#include <itkRawImageIO.h>

#include <sys/time.h>
#include <boost/filesystem.hpp>
#include <unistd.h>
#include <sys/stat.h>

#include "kdu_elementary.h"
#include "kdu_messaging.h"
#include "kdu_params.h"
#include "kdu_compressed.h"
#include "kdu_file_io.h"
#include "jp2.h"


#define BNB_RUN 0

using namespace std;
namespace fs = boost::filesystem;


const unsigned int Dimension = 2;
//typedef  unsigned char PixelType;
typedef  unsigned short PixelType;
typedef itk::Image<PixelType,Dimension> ImageType;
typedef itk::CenteredRigid2DTransform<double> TransformType;

//const string  outFolder("/data1/StackAlign/PORTALJP2");
const string  outFolder("/data/mitra2/PORTALJP2");


class kdu_stream_message : public kdu_message {
  public: // Member classes
    kdu_stream_message(std::ostream *stream)
      { this->stream = stream; }
    void put_text(const char *string)
      { (*stream) << string; }
    void flush(bool end_of_message=false)
      { stream->flush(); }
  private: // Data
    std::ostream *stream;
  };

static kdu_stream_message cout_message(&std::cout);
static kdu_stream_message cerr_message(&std::cerr);
static kdu_message_formatter pretty_cout(&cout_message);
static kdu_message_formatter pretty_cerr(&cerr_message);


class Xform{

private:
	int  hflip,vflip,sproc;
	double R, D[2], O[2], A[2], T[2],S[2];
	double spacing[2];
	itk::Matrix<double,2,2> direction;
	unsigned int parts;
	size_t outWidth, outHeight;
	
	string 	BrainName,
			SectionName,
			directory,
			tempfolder,
			thumbnailfname,
			ifname,
			ofname,
			temptiff;
	fs::path iJP2Path, oJP2Path, oJPGPath; 

	vector<string> rawFiles, xformFiles;
	TransformType::Pointer Tx;
	itk::Size<2> iSize;
	ImageType::PointType origin;

public:	
	Xform(const size_t, const size_t);
	bool parseParamline(string& line);
	bool adjustParameters();
	void readJP2Header();
	void cleanup();
	bool execute();
	void prepare();
	PixelType getDefPixelVal(ImageType::Pointer&);
};

Xform::Xform(const size_t gx, const size_t gy) {
	R = 0.0; 
	for (int i=0;i<2;++i){
		S[i] = O[i] = A[i] = T[i] = D[i] = 0.0;
	}
	outWidth = gx;
	outHeight = gy;
	parts = 4;
	
	iSize[0] = 0;
	iSize[1] = 0;
	
	tempfolder.assign("temp");
	stringstream t;
	srand (time(NULL));
	t << setw(4) << setfill('0') << rand()%9999;
	tempfolder.append(t.str());	
}

bool Xform::adjustParameters() {
	double asRatio = (double)iSize[0]/(double)iSize[1]; //x/y
	if (vnl_math_abs(asRatio*S[1] - S[0]) > 5) {
		return false;
	}
	double adjust = (double)iSize[0]/S[0] + (double)iSize[1]/S[1] ; 
	adjust *= 0.5;
	for (int i=0;i<2;++i) {
		T[i] *= adjust;
		O[i] *= adjust;
		A[i] *= adjust;
		origin[i] *= adjust;
	}
	cout << "Ratio:" << (double)iSize[0]/S[0] << "," << (double)iSize[1]/S[1] << endl;
	cout << "Adjusted parameters: T = [" << T[0] << "," << T[1] << "], O = [" << origin[0] << "," << origin[1] << "], A = [" << A[0] << "," << A[1] << "]" << endl;
	return true;
}

void Xform::readJP2Header() {
  kdu_customize_warnings(&pretty_cout);
  kdu_customize_errors(&pretty_cerr);

  // Construct code-stream object
  //kdu_simple_file_source input(argv[1]);
  jp2_source input;
  jp2_family_src jp2fmly;
  jp2fmly.open(iJP2Path.string().c_str());
  input.open(&jp2fmly);
  if (input.read_header() == true) {
	  std::cout << "Reading success" << std::endl;
  }
  kdu_codestream codestream; codestream.create(&input);
  codestream.set_fussy();
  kdu_dims dims; codestream.get_dims(0,dims);
  int num_components = codestream.get_num_components();
  if (num_components == 3)	{
      kdu_dims dims1; codestream.get_dims(1,dims1);
      kdu_dims dims2; codestream.get_dims(2,dims2);
      if ((dims1 != dims) || (dims2 != dims))	{
		  std::cout<<"Dimensions are inconsistent"<<std::endl;
	  }
  }
  else {
	  std::cout<<"Number of components are inconsistent"<<std::endl;
  }
  codestream.destroy();
  input.close(); // Not really necessary here.
  std::cout << "Extracted file of size [" << dims.size.x <<"," << dims.size.y <<"]" << std::endl << "Num colors " << num_components << std::endl;
  iSize[0] = dims.size.x;
  iSize[1] = dims.size.y;
}

void Xform::cleanup() {
	fs::path temppath = tempfolder;
	fs::remove_all(temppath);
}

bool Xform::parseParamline(string& line) {
	//"PMD789&788-F29-2012.09.23-04.30.55_PMD788_3_0087,/nfs/data/main/M8/mba_converted_imaging_data/PMD789&788/PMD788_JP2,-348.988,-140.406,0.969053,-0.246851,710,625,-1.2079,0,0,117.03,19.1169,0,0,0,"
	
	int state = 0;
	
	size_t n1 = 0, n2 = line.find_first_of(',');
	if (n2 == string::npos) {
		return false;
	}
	
	cout << "Reading parameters" << endl;
	while (n2 != string::npos) {
		switch(state) {
		case 0:
			SectionName = line.substr(n1,n2-n1); state++; break;
		case 1:
			directory = line.substr(n1,n2-n1); state++; break;
		case 2:
			istringstream(line.substr(n1,n2-n1)) >> O[0]; state++; break;
		case 3:	
			istringstream(line.substr(n1,n2-n1)) >> O[1]; state++; break;
		case 4:	
			istringstream(line.substr(n1,n2-n1)) >> D[0]; state++; break;
		case 5:
			istringstream(line.substr(n1,n2-n1)) >> D[1]; state++; break;
		case 6:
			istringstream(line.substr(n1,n2-n1)) >> S[0]; state++; break;
		case 7:	
			istringstream(line.substr(n1,n2-n1)) >> S[1]; state++; break;
		case 8:	
			istringstream(line.substr(n1,n2-n1)) >> R; state++; break;
		case 9:
			istringstream(line.substr(n1,n2-n1)) >> A[0]; state++; break;
		case 10:
			istringstream(line.substr(n1,n2-n1)) >> A[1]; state++; break;	
		case 11:
			istringstream(line.substr(n1,n2-n1)) >> T[0]; state++; break;
		case 12:
			istringstream(line.substr(n1,n2-n1)) >> T[1]; state++; break;			
		case 13:
			istringstream(line.substr(n1,n2-n1)) >> hflip; state++; break;
		case 14:
			istringstream(line.substr(n1,n2-n1)) >> vflip; state++; break;
		case 15:
			istringstream(line.substr(n1,n2-n1)) >> sproc; state++; break;			
		}
		n1=n2+1;
		n2 = line.find_first_of(',',n1);
	}
	
	cout << "Section: " << SectionName << endl << "Path: " << endl;
	cout << "R = " << R<<"("<<R*180.0/3.1416 <<")"<< endl 
	<<"T = [" << T[0] << "," << T[1] <<"]"<< endl 
	<<"A = [" << A[0] << "," << A[1] <<"]"<< endl 
	<<"S = [" << S[0] << "," << S[1] <<"]"<< endl 
	<<"D = [" << D[0] << "," << D[1] <<"]"<< endl 
	<<"O = [" << O[0] << "," << O[1] <<"]"<< endl 
	<<"Hflip = " << hflip << " Vflip = " << vflip << " Transpose = " << sproc << endl;
	
	return true;
}

void Xform::prepare() {
	BrainName.assign("UNKNOWN");
	//PMD789&788-F29-2012.09.23-04.30.55_PMD788_3_0087
	size_t n1 = SectionName.find_first_of("_");
	if (n1 != string::npos) {
		size_t n2 = SectionName.find_first_of("_",n1+1);
		if (n2 != string::npos) {
			BrainName = SectionName.substr(n1+1,n2-n1-1);
		}
	}
	
	//adjust path for bnb
	typedef pair<string,string> ElementType;
	vector <ElementType> MDrives;
	MDrives.push_back(ElementType("/nfs/data/main/M1/","/nfs/mitraweb2/mnt/disk020/main/")); //(M1)
	MDrives.push_back(ElementType("/nfs/data/main/M2/","/nfs/mitraweb2/mnt/disk021/main/")); //(M2)
	MDrives.push_back(ElementType("/nfs/data/main/M3/","/nfs/mitraweb2/mnt/disk022/main/")); //(M3)
	MDrives.push_back(ElementType("/nfs/data/main/M4/","/nfs/mitraweb2/mnt/disk023/main/")); //(M4)
	MDrives.push_back(ElementType("/nfs/data/main/M5/","/nfs/mitraweb2/mnt/disk024/main/")); //(M5)
	MDrives.push_back(ElementType("/nfs/data/main/M6/","/nfs/mitraweb2/mnt/disk030/main/")); //(M6)
	MDrives.push_back(ElementType("/nfs/data/main/M7/","/nfs/mitraweb2/mnt/disk031/main/")); //(M7)
	MDrives.push_back(ElementType("/nfs/data/main/M8/","/nfs/mitraweb2/mnt/disk032/main/")); //(M8)
	MDrives.push_back(ElementType("/nfs/data/main/M9/","/nfs/mitraweb2/mnt/disk033/main/")); //(M9)
	MDrives.push_back(ElementType("/nfs/data/main/M10/","/nfs/mitraweb2/mnt/disk034/main/")); //(M10)
	MDrives.push_back(ElementType("/nfs/data/main/M11/","/nfs/mitraweb2/mnt/disk080/main/")); //(M11)
	MDrives.push_back(ElementType("/nfs/data/main/M12/","/nfs/mitraweb2/mnt/disk081/main/")); //(M12)
	MDrives.push_back(ElementType("/nfs/data/main/M13/","/nfs/mitraweb2/mnt/disk082/main/")); //(M13)
	MDrives.push_back(ElementType("/nfs/data/main/M14/","/nfs/mitraweb2/mnt/disk083/main/")); //(M14)
	MDrives.push_back(ElementType("/nfs/data/main/M15/","/nfs/mitraweb2/mnt/disk084/main/")); //(M15)
	MDrives.push_back(ElementType("/nfs/data/main/M16/","/nfs/mitraweb2/mnt/disk100/main/")); //(M16)
	MDrives.push_back(ElementType("/nfs/data/main/M17/","/nfs/mitraweb2/mnt/disk101/main/")); //(M17)
	MDrives.push_back(ElementType("/nfs/data/main/M18/","/nfs/mitraweb2/mnt/disk102/main/")); //(M18)
	MDrives.push_back(ElementType("/nfs/data/main/M19/","/nfs/mitraweb2/mnt/disk103/main/")); //(M19)
	MDrives.push_back(ElementType("/nfs/data/main/M20/","/nfs/mitraweb2/mnt/disk104/main/")); //(M20)
	
	for (unsigned int i = 0; i<MDrives.size(); ++i) {
		if (directory.find(MDrives[i].first) != string::npos) {
			directory.replace(0,MDrives[i].first.length(),MDrives[i].second);	
			break;	
		}
	}

	
	fs::path BrainPath = outFolder;
	BrainPath /= BrainName;
	if (fs::exists(BrainPath) == false) {
		fs::create_directory(BrainPath);
	}
	
	ifname = SectionName + "_lossless.jp2";
	iJP2Path = directory;
	//iJP2Path += "_JP2";
	iJP2Path /= ifname;

	ofname = SectionName + ".jp2";
	thumbnailfname = SectionName + ".jpg";
	
	oJP2Path = outFolder;
	oJP2Path /= BrainName;
	oJP2Path /= ofname;
	oJPGPath = outFolder;
	oJPGPath /= BrainName;
	oJPGPath /= thumbnailfname;
	
	cout << "Input " << iJP2Path.string() << endl << "Output " << oJP2Path.string() << endl;
	

	//fs::path temppath = fs::temp_directory_path();
	//temppath /= tempfolder;
	fs::path temppath = tempfolder;
	if (fs::exists(temppath) == false) {
		fs::create_directory(temppath);
	}
	tempfolder = temppath.string();
	
	rawFiles.push_back(tempfolder + string("/RED.rawl"));
	rawFiles.push_back(tempfolder + string("/GREEN.rawl"));
	rawFiles.push_back(tempfolder + string("/BLUE.rawl"));
	xformFiles.push_back(tempfolder + string("/XFRED.rawl"));
	xformFiles.push_back(tempfolder + string("/XFGREEN.rawl"));
	xformFiles.push_back(tempfolder + string("/XFBLUE.rawl"));
	
	temptiff = tempfolder + string("/temp.tif");

	origin[0] = O[0];
	origin[1] = O[1];
	
	direction(0,0) = D[0];
	direction(0,1) = D[1];
	direction(1,0) = -1.0*D[1];
	direction(1,1) = D[0];
	
	spacing[0] = 1.0;
	spacing[1] = 1.0;

}

PixelType Xform::getDefPixelVal(ImageType::Pointer& img) {
	//double val = 0.0, count = 0.0;
	vector<PixelType> valList(10000);
	size_t h = 100;
	for (size_t y=0; y<img->GetBufferedRegion().GetSize(1);++y) {
		for (size_t x=0; x<h;++x) {
			itk::Index<2> ndx = {{x,y}};
			PixelType m = img->GetPixel(ndx);
			if ((m != 0) | (m != 255)) {
				//val += static_cast<double>(m)/255.0;
				//count += 1.0;
				valList.push_back(m);
			}
			ndx[0] = img->GetBufferedRegion().GetSize(0)-1-x;
			m = img->GetPixel(ndx);
			if ((m != 0) | (m != 255)) {
				//val += static_cast<double>(m)/255.0;
				//count += 1.0;
				valList.push_back(m);
			}
		}
	}
	for (size_t x=0; x<img->GetBufferedRegion().GetSize(0);++x) {
		for (size_t y=0; y<h;++y) {
			itk::Index<2> ndx = {{x,y}};
			PixelType m = img->GetPixel(ndx);
			if ((m != 0) | (m != 255)) {
				//val += static_cast<double>(m)/255.0;
				//count += 1.0;
				valList.push_back(m);
			}
			ndx[1] = img->GetBufferedRegion().GetSize(1)-1-y;
			m = img->GetPixel(ndx);
			if ((m != 0) | (m != 255)) {
				//val += static_cast<double>(img->GetPixel(ndx))/255.0;
				//count += 1.0;
				valList.push_back(m);
			}
		}
	}
	
	sort(valList.begin(),valList.end());
	size_t midpt = valList.size()/2;
	return valList.at(midpt);
	//return static_cast<PixelType>(255*val/count);
}



bool Xform::execute() {
	
	Tx = TransformType::New();
	Tx->SetAngle(R);
	TransformType::OutputVectorType trans;
	trans[0] = T[0];
	trans[1] = T[1];
	Tx->SetTranslation(trans);
	TransformType::OutputPointType center;
	center[0] = A[0];
	center[1] = A[1];
	Tx->SetCenter(center);
	
	//Tx->SetIdentity();
	boost::system::error_code ec;
	fs::path rawJP2("original.jp2");
	fs::copy_file(iJP2Path,rawJP2,fs::copy_option::overwrite_if_exists,ec);
	if (ec.value() != 0) {
		cout << "FAIL copying input file:" << ec.message() << endl;
		return false;
	}	
	//uncompress lossless Jp2 file into RGB components
	stringstream jp2expcmd;
	jp2expcmd << "kdu_expand -i original.jp2 -o " << rawFiles[0] << "," << rawFiles[1] << "," << rawFiles[2] << " -resilient" << endl;
	//jp2expcmd << "kdu_expand -i \"" << iJP2Path.string() << "\" -o " << rawFiles[0] << "," << rawFiles[1] << "," << rawFiles[2] << " -resilient" << endl;
	cout << jp2expcmd.str() << endl;
	int status = system(jp2expcmd.str().c_str());
	cout << "JP2expand: " << status << endl;
	if (status != 0) { 
		return false; 
	}
	
	fs::remove(rawJP2);
	
	//for each component,
	//load component totally into memory.
	//output Partially, and write to disk.
	//keep appending unil all parts are created.
	//kdu_compress.
	for (int comp=0;  comp<3; ++comp) {
		
		typedef itk::RawImageIO <PixelType,2> ReaderType;
		ReaderType::Pointer io = ReaderType::New();
		io->SetByteOrderToLittleEndian();
		io->SetFileTypeToBinary();
		io->SetDimensions(0,iSize[0]);
		io->SetDimensions(1,iSize[1]);

		itk::ImageFileReader<ImageType>::Pointer reader = itk::ImageFileReader<ImageType>::New();
		reader->SetImageIO(io);
		reader->SetFileName(rawFiles[comp]);
		try {
			reader->Update();
		}
		catch (itk::ExceptionObject e) {
			e.Print(cout);
			return false;
		}
		ImageType::Pointer img = reader->GetOutput();
		img->SetOrigin(origin);
		img->SetSpacing(spacing);
		img->SetDirection(direction);
		cout << "Input image spans " << origin[0] << " to " << origin[0]+(double)iSize[0] << " , " <<
							origin[1] << " to " << origin[1]+(double)iSize[1] << endl;
				
		//use FLIP here if necessary
		
		//border adjustment
		
		//special processing
		if (sproc == 1) {
			//rescale intensities by 16
			ImageType::PixelType BITSHIFT = 8;
			itk::ImageRegionIterator<ImageType> it1(img,img->GetBufferedRegion());
			for (it1.GoToBegin(); !it1.IsAtEnd(); ++it1) {
				it1.Set(it1.Get()/BITSHIFT);
			}
		}
		
		typedef itk::BSplineInterpolateImageFunction<ImageType,double,double> InterpolatorType;
		InterpolatorType::Pointer resampler = InterpolatorType::New();
		typedef itk::NearestNeighborExtrapolateImageFunction<ImageType, double> ExtrapolatorType;
		ExtrapolatorType::Pointer extrapolator = ExtrapolatorType::New();
		typedef itk::ResampleImageFilter< ImageType, ImageType >    ResampleFilterType;
		ResampleFilterType::Pointer resample = ResampleFilterType::New();
		resample->SetTransform( Tx );
		resample->SetInput( img );
		resample->SetInterpolator(resampler);
		//resample->SetExtrapolator(extrapolator);
		resample->SetDefaultPixelValue(getDefPixelVal(img));
		ImageType::SizeType sz;

		//outWidth, outHeight
		size_t npixel, height, width, cumHeight;
		width = outWidth;
		height = outHeight/parts;
		double start[2], startXformed[2];
		start[0] = -1.0*double(outWidth)/2.0 ;
		start[1] = -1.0*double(outHeight)/2.0 ;
		startXformed[0] = D[0]*start[0] - D[1]*start[1];
		startXformed[1] = D[1]*start[0] + D[0]*start[1]; 
		
		cout << "Output image spans " << startXformed[0] << " to " << startXformed[0]+(double)width << " , " <<
							startXformed[1] << " to " << startXformed[1]+(double)height << endl;
		
		cumHeight = 0;

		//erase the file
		ofstream fid(xformFiles[comp].c_str(),ios::binary);
		fid.close();
		ImageType::PointType orig;
		
		for (unsigned int m = 0; m < parts; ++m)	{

			sz[0] = width;
			sz[1] = vnl_math_min(height,outHeight-cumHeight+height);
			npixel = sz[0]*sz[1];
		
			orig[0] = startXformed[0];
			orig[1] = startXformed[1] + (double)cumHeight;
			cumHeight += sz[1];

			cout <<" Y sweeping from "<<orig[1]<<" to "<< orig[1]+(double)sz[1]<< endl;

			resample->SetSize( sz );
			resample->SetOutputOrigin( orig );
			resample->SetOutputSpacing( spacing );
			//resample->SetDefaultPixelValue( 0 );
			resample->Update();
			ofstream fid(xformFiles[comp].c_str(),ios::binary | ios::app);
			if (fid.is_open() == false) {
				return false;	
			}
			fid.write((char*)resample->GetOutput()->GetBufferPointer(),npixel*sizeof(PixelType)).flush();
			fid.close();
		}
		cout << "Complete component " << comp << endl;
		//cin.get();
	}
	
	fs::remove(fs::path(rawFiles[0]));
	fs::remove(fs::path(rawFiles[1]));
	fs::remove(fs::path(rawFiles[2]));
	stringstream jp2cmpcmd;
	if (sizeof(PixelType) == 1) {
		cout << "Compressing 8 bit " << endl;
		jp2cmpcmd << "kdu_compress -i " << xformFiles[0] <<"," << xformFiles[1] << "," << xformFiles[2] << " -o \"" << oJP2Path.string() <<"\" Sprecision=8 Ssigned=no Sdims=\\{" 
			<<  outHeight << "," << outWidth <<	"\\} -rate 1 Qstep=0.00001 Clevels=20 Clayers=8 ORGgen_plt=yes ORGtparts=R Cblk=\\{32,32\\} ";
	}
	else {
		cout << "Compressing 16 bit " << endl;
		jp2cmpcmd << "kdu_compress -i " << xformFiles[0] <<"," << xformFiles[1] << "," << xformFiles[2] << " -o \"" << oJP2Path.string() <<"\" Sprecision=16 Ssigned=no Sdims=\\{" 
			<<  outHeight << "," << outWidth <<	"\\} -rate 1 Qstep=0.00001 Clevels=20 Clayers=8 ORGgen_plt=yes ORGtparts=R Cblk=\\{32,32\\} ";
	}
	cout << jp2cmpcmd.str() << endl;
	status = system(jp2cmpcmd.str().c_str());
	cout << "JP2 Lossy: " << status << endl;
	if (status != 0) {return false;}

	fs::remove(fs::path(xformFiles[0]));
	fs::remove(fs::path(xformFiles[1]));
	fs::remove(fs::path(xformFiles[2]));
	//write jpeg image
	
	stringstream jpegcmd;
	jpegcmd << "kdu_expand -i \"" << oJP2Path.string() << "\" -o \"" << temptiff << "\" -reduce 6 " << endl;
	cout << jpegcmd.str() << endl;
	cout << "jpegfile: " << system(jpegcmd.str().c_str()) << endl;
	
	typedef itk::RGBPixel<PixelType> InputRGBPixelType;
	typedef itk::Image<InputRGBPixelType, Dimension> InputRGBImageType;
	typedef itk::RGBPixel<unsigned char> OutputRGBPixelType;
	typedef itk::Image<OutputRGBPixelType, Dimension> OutputRGBImageType;

	itk::ImageFileReader<InputRGBImageType>::Pointer threader = itk::ImageFileReader<InputRGBImageType>::New();
	threader->SetFileName(temptiff);
	InputRGBImageType::Pointer thumbnailfull = threader->GetOutput();
	try {thumbnailfull->Update();}
	catch(itk::ExceptionObject e) { e.Print(cout); return false;}
	OutputRGBImageType::Pointer thumbnail = OutputRGBImageType::New();
	itk::Size<2> thsz, thfullsz;
	thfullsz = thumbnailfull->GetBufferedRegion().GetSize();
	thsz[0] = 120; thsz[1] = 90;
	thumbnail->SetRegions(thsz);
	thumbnail->Allocate();
	size_t ity, itx;
	
	itk::ImageRegionIterator<InputRGBImageType> it(thumbnailfull,thumbnailfull->GetBufferedRegion().GetSize());
	vector<PixelType> normVal; 
	normVal.reserve(thfullsz[0]*thfullsz[1]*3);
	for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
		InputRGBPixelType value = it.Get();
		normVal.push_back(value[0]);
		normVal.push_back(value[1]);
		normVal.push_back(value[2]);
	}
	sort(normVal.begin(),normVal.end());
	size_t lenLow = normVal.size()/120;
	size_t lenHigh = normVal.size() - lenLow;
	PixelType lowLimit = normVal[lenLow];
	PixelType highLimit = normVal[lenHigh];
	if (highLimit < 255) {highLimit = 255;}
	//PixelType factor = 255/(highLimit - lowLimit);
	cout << "Normalizing factors : " << highLimit << " " << lowLimit << endl;	
		
	
	for (size_t ty=0; ty<90; ++ty){
		ity = (ty*thfullsz[1])/90;
		for (size_t tx=0; tx<120; ++tx) {
			itx = (tx*thfullsz[0])/120;
			itk::Index<2> ndxfull = {{itx, ity}};
			itk::Index<2> ndx = {{tx, ty}};
			InputRGBPixelType value = thumbnailfull->GetPixel(ndxfull);
			OutputRGBPixelType oval;
			for (int i=0; i<3; ++i) {
				if (value[i] < lowLimit) {
					oval[i] = 0;
				}
				else if (value[i] > highLimit) {
					oval[i] = 255;
				}
				else {
					oval[i] = static_cast<unsigned char>(((value[i]-lowLimit)*255)/(highLimit - lowLimit));
				}
			}
			thumbnail->SetPixel(ndx,oval);;
		}
	}
	
	
	itk::ImageFileWriter<OutputRGBImageType>::Pointer thwriter = itk::ImageFileWriter<OutputRGBImageType>::New();
	thwriter->SetFileName(oJPGPath.string());
	thwriter->SetInput(thumbnail);
	thwriter->Update();
	cout << "Thumbnail written in " << oJPGPath.string() << endl;
	fs::remove(fs::path(temptiff));

	return true;
}


int main( int argc, char * argv[] ) {
	
	if (argc != 2) {
		cout << "Usage: paramLine " <<	endl;
		cout << "Supplied " << argc << " arguements" << endl; 
		return EXIT_FAILURE;
	}

	string param(argv[1]);

	//string param("PMD789&788-F26-2011.09.09-19.16.00_PMD788_1_0076,/nfs/data/main/M8/mba_converted_imaging_data/PMD789&788/PMD788_JP2,-328.08,-169.625,0.969053,-0.246851,661,516,-0.0372065,-28.194,-17.092,-72.439,93.4285,0,0,0,");
	
	const size_t gx = 24000;
	const size_t gy = 18000;

	Xform xf(gx,gy);

	if (xf.parseParamline(param) == false) {
		cout << "Cannot parse parameters " << param << endl;
		return EXIT_FAILURE;
	}
	
	xf.prepare(); //get the JP2 filename
	xf.readJP2Header(); //get filesize
	
	if (xf.adjustParameters() == false) {
		cout << "Parameters not matching the image" << endl;
		return EXIT_FAILURE;
	}
	

	if (xf.execute() == false) {
		 return EXIT_FAILURE;
	}
	xf.cleanup();

	return EXIT_SUCCESS;
}

