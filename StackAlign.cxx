#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <ctime>
#include <cmath>
#include <vector>
#include <iomanip>
#include <boost/algorithm/string.hpp>
//#include <sys/time.h>
#include <boost/filesystem.hpp>
//#include <unistd.h>
//#include <sys/stat.h>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkCenteredTransformInitializer.h"
#include "itkMedianImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkCenteredRigid2DTransform.h"
#include "itkSubtractImageFilter.h"

#include "itkImageRegistrationMethod.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkMultiResolutionImageRegistrationMethod.h"
#include "itkCheckerBoardImageFilter.h"


#include "itkArray.h"
#define DEBUGPRINT(x) if (debugON==true) {cout << x << endl;};


using namespace std;
namespace fs = boost::filesystem;

const unsigned int Dimension = 2;
typedef  float PixelType;
bool debugON;
fs::path DebugDirectory;
int debugXformCount = 0;

typedef itk::Image<unsigned char,Dimension> CharImageType;
typedef itk::Image<PixelType,Dimension> ImageType;
typedef itk::RGBPixel<PixelType> RGBPixelType;
typedef itk::Image<RGBPixelType, Dimension> RGBImageType;
typedef itk::RGBPixel<unsigned short> RGBPixelType16;
typedef itk::Image<RGBPixelType16, Dimension> RGBImageType16;
typedef itk::RGBPixel<unsigned char> RGBPixelType8;
typedef itk::Image<RGBPixelType8, Dimension> RGBImageType8;
typedef itk::CenteredRigid2DTransform<double> TransformType;

void WriteDiffImage(const char* filename, ImageType::Pointer&fimg, ImageType::Pointer&mimg,TransformType::Pointer& Tx, int gx = 1000, int gy = 1000) {
	//transforms the moving image (mimg) by Tx and superimposes on fixed image (fimg)
	itk::RescaleIntensityImageFilter<ImageType,CharImageType>::Pointer rescaler1 = itk::RescaleIntensityImageFilter<ImageType,CharImageType>::New();
	rescaler1->SetOutputMinimum(0);
	rescaler1->SetOutputMaximum(127);
	rescaler1->SetInput(fimg);
	CharImageType::Pointer fimg2 = rescaler1->GetOutput();
	fimg2->Update();

	itk::RescaleIntensityImageFilter<ImageType,CharImageType>::Pointer rescaler2 = itk::RescaleIntensityImageFilter<ImageType,CharImageType>::New();
	rescaler2->SetOutputMinimum(0);
	rescaler2->SetOutputMaximum(127);
	rescaler2->SetInput(mimg);

	typedef itk::ResampleImageFilter<CharImageType, CharImageType>  ResampleFilterType;
	ResampleFilterType::Pointer resample = ResampleFilterType::New();
	resample->SetTransform( Tx );
	resample->SetInput( rescaler2->GetOutput() );	
	ImageType::SizeType sz;
	sz[0] = gx;
	sz[1] = gy;
	double o1[2];
	o1[0] = static_cast<double>(-gx/2);
	o1[1] = static_cast<double>(-gy/2);
	resample->SetSize( sz );
	resample->SetOutputOrigin( o1 );
	double spacing[2]  = {1.0, 1.0};
	resample->SetOutputSpacing( spacing );
	resample->SetDefaultPixelValue( 127 );
	CharImageType::Pointer mimg2 = resample->GetOutput();
	mimg2->Update();
	
	itk::CheckerBoardImageFilter<CharImageType>::Pointer checkerbd =  itk::CheckerBoardImageFilter<CharImageType>::New();
	checkerbd->SetInput1(fimg2);
	checkerbd->SetInput2(mimg2);

	itk::ImageFileWriter<CharImageType>::Pointer writer = itk::ImageFileWriter<CharImageType>::New();
	writer->SetInput(checkerbd->GetOutput());
	stringstream Fname;
	Fname << filename << ".png";
	writer->SetFileName(Fname.str());
	writer->Update();
	/*
	CharImageType::Pointer diff = CharImageType::New();
	diff->SetRegions(sz);
	diff->Allocate();
	diff->FillBuffer(0);
	
	itk::ImageRegionIterator<CharImageType> itf(fimg2,fimg2->GetBufferedRegion().GetSize());
	itk::ImageRegionIterator<CharImageType> itm(mimg2,mimg2->GetBufferedRegion().GetSize());
	itk::ImageRegionIterator<CharImageType> itd(diff,diff->GetBufferedRegion().GetSize());
	for (itf.GoToBegin(), itm.GoToBegin(), itd.GoToBegin(); !itf.IsAtEnd(); ++itf, ++itd, ++itm) {
		itd.Set(128 + itf.Get() - itm.Get());
	}
	
	itk::ImageFileWriter<CharImageType>::Pointer writer = itk::ImageFileWriter<CharImageType>::New();
	writer->SetInput(diff);
	stringstream Fname;
	Fname << filename << ".png";
	writer->SetFileName(Fname.str());
	writer->Update();
	*/
}




void WriteImage(const char* filename, ImageType::Pointer&img, TransformType::Pointer& Tx, int gx = 1000, int gy = 1000) {

	itk::RescaleIntensityImageFilter<ImageType,CharImageType>::Pointer rescaler = itk::RescaleIntensityImageFilter<ImageType,CharImageType>::New();
	rescaler->SetOutputMinimum(0);
	rescaler->SetOutputMaximum(255);
	rescaler->SetInput(img);

	typedef itk::ResampleImageFilter<CharImageType, CharImageType>  ResampleFilterType;
	ResampleFilterType::Pointer resample = ResampleFilterType::New();
	resample->SetTransform( Tx );
	resample->SetInput( rescaler->GetOutput() );	
	ImageType::SizeType sz;
	sz[0] = gx;
	sz[1] = gy;
	double o1[2];
	o1[0] = static_cast<double>(-gx/2);
	o1[1] = static_cast<double>(-gy/2);
	resample->SetSize( sz );
	resample->SetOutputOrigin( o1 );
	double spacing[2]  = {1.0, 1.0};
	resample->SetOutputSpacing( spacing );
	resample->SetDefaultPixelValue( 127 );

	itk::ImageFileWriter<CharImageType>::Pointer writer = itk::ImageFileWriter<CharImageType>::New();
	writer->SetInput(resample->GetOutput());
	stringstream grayFname;
	grayFname << filename << ".png";
	//cout << "Writing intermediate file: " << grayFname.str() << endl;
	writer->SetFileName(grayFname.str());
	writer->Update();
}


void WriteImage(const char* filename, ImageType::Pointer&img, int gx = 1000, int gy = 1000) {
	TransformType::Pointer Tx = TransformType::New();
	Tx->SetIdentity();
	WriteImage(filename, img, Tx, gx, gy);
}


class Section {
public:
	string SectionName;
	fs::path SectionPath;
	unsigned int SequenceNumber;
	bool Usable;
	bool active;
	//ImageType::Pointer img;
	TransformType::Pointer Tx;
	Section *next, *prev;
	int ndx;
	ImageType::SizeType size;
	bool hflip, vflip, spProc;
	ImageType::PointType origin, shift;
	double error;
	itk::Vector<double,2> direction;
	
	Section();
	void Show();
	void initImageCenterFromMoment();
	bool getPrepImage(double beta_inv, ImageType::Pointer& img);
	void Print() {cout << "Usable: " << (int)Usable << "("<< SequenceNumber <<") SectionName: " << SectionName <<  endl;}
	
};

class Links {
public:
	Section *sec1;	
	Section *sec2;
	TransformType::Pointer Tx12;
	double error;
	Links() {};
	Links(Section*s1, Section*s2, TransformType::Pointer& t, double e) {
		sec1 = s1;
		sec2 = s2;
		Tx12 = t;
		error = e;
	}
	void Print() {
		if (sec1 != NULL) {
			cout << "Sec1: " << sec1->SequenceNumber;
		}
		else {cout << "XXX";}
		cout << " - ";
		if (sec2 != NULL) {
			cout << "Sec2: " << sec2->SequenceNumber ;
		}
		else {cout << "XXX";}
		cout << endl;
		if (Tx12) {
			cout << Tx12->GetParameters() << endl;
		}
	}
};

Section::Section() {
	this->Tx = TransformType::New();
	this->Tx->SetIdentity();
	this->error = 10000.0;
	this->Usable = false;
	this->next = this->prev = NULL;
	//this->img = NULL;
	this->ndx = -1;
	this->SequenceNumber = 0;
	this->SectionName.assign("NameNotAvailable");
	this->SectionPath = "PathNotAvailable";
	for (int i=0; i<2; ++i) {
		this->origin[i] = 0.0;
		this->direction[i] = 0.0;
		this->size[i] = 0;
	}
	this->direction[0] = 1.0;
}

void Section::Show() {
	cout << "Sec: "  << SectionName << endl 
	<< "Path: " << SectionPath << endl
	<< "SequenceIndex: " << SequenceNumber <<
	origin << "," << direction << "," <<
	size << endl;
}


bool Section::getPrepImage(double beta_inv, ImageType::Pointer& img) {
	itk::ImageFileReader<RGBImageType>::Pointer reader = itk::ImageFileReader<RGBImageType>::New();
	reader->SetFileName(this->SectionPath.string());
	RGBImageType::Pointer imRGB = reader->GetOutput();
	DEBUGPRINT("Reading " << SectionPath.string())
	try {
		imRGB->Update();
	}

	catch (itk::ExceptionObject e) {
		cout << "Error in reading" << this->SectionPath.string() << endl << e.GetDescription() << endl;
		return false;
	}
	this->size = imRGB->GetBufferedRegion().GetSize();
	cout << this->SectionPath.string() << " read into memory of size " << imRGB->GetBufferedRegion().GetSize() << endl;
	itk::ImageRegionIterator<RGBImageType> it1(imRGB,imRGB->GetBufferedRegion());

	img = ImageType::New();
	img->SetRegions(imRGB->GetBufferedRegion());
	img->Allocate();
	double spacing[2] = {1.0,1.0};
	img->SetSpacing(spacing);

	itk::ImageRegionIterator<ImageType> it2(img,img->GetBufferedRegion());
	
	CharImageType::Pointer mask = CharImageType::New();
	mask->SetRegions(imRGB->GetBufferedRegion());
	mask->Allocate();
	mask->FillBuffer(0);
	itk::ImageRegionIterator<CharImageType> mit(mask,mask->GetBufferedRegion());
	for(it1.GoToBegin(), mit.GoToBegin(); !mit.IsAtEnd(); ++it1, ++mit) {
		RGBPixelType rgb = it1.Get();
		if ( ((rgb[0]<=255) && (rgb[1]<=255) && (rgb[2]<=255) && (rgb[0]>252) && (rgb[1]>252) && (rgb[2]>252)) ||
			((rgb[0]<3) && (rgb[1]<3) && (rgb[2]<3)) ) {
			mit.Set(255);
		}
	}
	
	PixelType pixMax = 0.0, pixMin = 255.0*255.0;
	for (int col = 0; col<3; ++col) {
		//cout << "Color " << col << endl;
		for(it1.GoToBegin(), it2.GoToBegin(); !it1.IsAtEnd(); ++it1, ++it2) {
			it2.Set(it1.Get()[col]);
			if (pixMax < it1.Get()[col]) {pixMax = it1.Get()[col];}
			if (pixMin > it1.Get()[col]) {pixMin = it1.Get()[col];}
		}

		itk::MedianImageFilter<ImageType,ImageType>::Pointer medFilter = itk::MedianImageFilter<ImageType,ImageType>::New();
		itk::MedianImageFilter<ImageType,ImageType>::RadiusType rad; rad.Fill(1);
		medFilter->SetRadius(rad);
		
		//itk::DiscreteGaussianImageFilter<ImageType,ImageType>::Pointer gaussFilter = itk::DiscreteGaussianImageFilter<ImageType,ImageType>::New();
		//gaussFilter->SetVariance(9.0);

		itk::SmoothingRecursiveGaussianImageFilter<ImageType,ImageType>::Pointer rgaussFilter = itk::SmoothingRecursiveGaussianImageFilter<ImageType,ImageType>::New();
		rgaussFilter->SetSigma(2.0);
		rgaussFilter->SetNormalizeAcrossScale(true);
		
		itk::MedianImageFilter<ImageType,ImageType>::Pointer medFilter2 = itk::MedianImageFilter<ImageType,ImageType>::New();
		medFilter2->SetRadius(rad);
		
		medFilter->SetInput(img);
		rgaussFilter->SetInput(medFilter->GetOutput());
		//gaussFilter->SetInput(respimg);
		medFilter2->SetInput(rgaussFilter->GetOutput());
		//medFilter2->SetInput(medFilter->GetOutput());
		ImageType::Pointer im2 = medFilter2->GetOutput();
		im2->Update();

		itk::ImageRegionIterator<ImageType> it3(im2,im2->GetBufferedRegion());
		
		for(it1.GoToBegin(), it3.GoToBegin(); !it1.IsAtEnd(); ++it1, ++it3) {
			it1.Value()[col] = it3.Get();
		}
	}

	if ((pixMax - pixMin) < 10) {
		cout << "Dynamic range of the data bad " << pixMax << " " << pixMin << endl;
		for (it2.GoToBegin(); !it2.IsAtEnd(); ++it2) {
			it2.Set(0.0);
		}
		//cin.get();
		return false;
	}	

	//get boundaries and background estimates, read RGB and update R,G,B, count + respimg

	unsigned int h = 20;
	vector<double> R, G, B;
	R.reserve(imRGB->GetBufferedRegion().GetSize()[0]*imRGB->GetBufferedRegion().GetSize()[1]);
	G.reserve(imRGB->GetBufferedRegion().GetSize()[0]*imRGB->GetBufferedRegion().GetSize()[1]);
	B.reserve(imRGB->GetBufferedRegion().GetSize()[0]*imRGB->GetBufferedRegion().GetSize()[1]);
	int count;
	for (int k=0; k<5; ++k) {
		R.clear();
		G.clear();
		B.clear();
		count = 0;
		for (mit.GoToBegin(), it1.GoToBegin(), it2.GoToBegin(); !it1.IsAtEnd(); ++it1, ++it2, ++mit) {
			
			if (mit.Get() > 0) {
				continue;
			}
			RGBPixelType rgb = it1.Get();
			itk::Index<2> ndx = it1.GetIndex();			
			if ((ndx[0] >= (unsigned int)(imRGB->GetBufferedRegion().GetSize()[0] - h)) || (ndx[0] < (unsigned int)h) ||
				(ndx[1] >= (unsigned int)(imRGB->GetBufferedRegion().GetSize()[1] - h)) || (ndx[1] < (unsigned int)h) ) {
				
				R.push_back((double)rgb[0]);
				G.push_back((double)rgb[1]);
				B.push_back((double)rgb[2]);
				count++;
			}
		}
		if (count > 1000.0) {
			break;
		}
		h += 20;
	}

	count /=2;
	sort(R.begin(),R.end());
	sort(G.begin(),G.end());
	sort(B.begin(),B.end());
	double mR = R[count];
	double mG = G[count];
	double mB = B[count];
	

	DEBUGPRINT( "R = " << mR << ", G = " << mG << ", B = " << mB )
	
	vector<double> dist;
	//this->img->FillBuffer(0.0);
	dist.reserve(imRGB->GetBufferedRegion().GetNumberOfPixels());
	for (mit.GoToBegin(), it1.GoToBegin(), it2.GoToBegin(); !it1.IsAtEnd(); ++mit, ++it1, ++it2) {
		if (mit.Get() == 0) {
			RGBPixelType rgb = it1.Get();
			double d = ((mR - (double)rgb[0])*(mR - (double)rgb[0])) + ((mG - (double)rgb[1])*(mG - (double)rgb[1])) + ((mB - (double)rgb[2])*(mB - (double)rgb[2]));
			d =  vcl_sqrt (d );
			dist.push_back( d );
			it2.Set(static_cast<float>(d));
		}
		else {
			it2.Set(0.0);
		}
	}
	
	sort(dist.begin(),dist.end());
	unsigned int midloc = dist.size()/2;
	double medDist = dist.at(midloc);
	
	DEBUGPRINT( "Data Size: " << dist.size() << " Midloc: " << midloc << endl << "Median: " << medDist )
	
	double fgmean = 0.0, fcount = 0.0;
	for ( vector<double>::iterator it5 = dist.begin(); it5 != dist.end(); ++it5) {
		if ((*it5) > medDist) {
			fgmean += vcl_log(*it5);
			fcount += 1.0;
		}
	}

	fgmean /= fcount;
	fgmean = vcl_exp(fgmean); 
	DEBUGPRINT( "Fg mean: " << fgmean );
	const double beta = 1.0/(beta_inv+0.001);
	it2.GoToBegin();
	unsigned int m = 0;
	for (mit.GoToBegin(),it2.GoToBegin(); !it2.IsAtEnd(); ++it2, ++mit) {
		if (mit.Get() == 0) {
			double d = beta*((double)it2.Get() - fgmean);
			d = 1.0/(1.0 + vcl_exp(-1.0*d));
			it2.Set(static_cast<float>(d));
			dist[m] = d;
			m++;
		}
	}
	sort(dist.begin(),dist.end());
	unsigned int cut1 = m/20;
	unsigned int cut2 = m - cut1;
	DEBUGPRINT( "Data Size: " << m << " Cut 1: " << dist[cut1] <<  " Cut 2: " << dist[cut2] )
	
	for (it2.GoToBegin(); !it2.IsAtEnd(); ++it2) {
		if(it2.Get() < dist[cut1]) {
			it2.Set(0.0);
		}
		else if (it2.Get() > dist[cut2]) {
			it2.Set(1.0);
		}
		else {
			PixelType v = (it2.Get()  - dist[cut1]) / (dist[cut2] - dist[cut1]);
			it2.Set(v) ;
			//cout << v << endl;
		}
	}
	
	itk::Size<2> size = img->GetBufferedRegion().GetSize();
	itk::ImageRegionIteratorWithIndex<ImageType> itc(img, img->GetBufferedRegion());
	
	double mX = 0.0, mY = 0.0, den = 0.0;
	for( itc.GoToBegin(); !itc.IsAtEnd(); ++itc) {
		double d = static_cast<double>(itc.Get());
		itk::Index<2> n = itc.GetIndex();
		double x = static_cast<double>(n[0])/static_cast<double>(size[0]) - 0.5;
		double y = static_cast<double>(n[1])/static_cast<double>(size[1]) - 0.5;
		mX += d*x;
		mY += d*y;
		den += d;
	}
	mX = 2.0*(mX/den)*static_cast<double>(size[0]);
	mY = 2.0*(mY/den)*static_cast<double>(size[1]);
	
	ImageType::PointType orig;
	orig[0] = -mX - (static_cast<double>(size[0])/2.0);
	orig[1] = -mY - (static_cast<double>(size[1])/2.0);
	img->SetOrigin(orig);
	this->origin = orig;
	DEBUGPRINT( "mX = " << orig[0] << ", mY = " << orig[1] << endl)
	
	if (debugON == true) {
		itk::RescaleIntensityImageFilter<ImageType,CharImageType>::Pointer rescaler = itk::RescaleIntensityImageFilter<ImageType,CharImageType>::New();
		rescaler->SetOutputMinimum(0);
		rescaler->SetOutputMaximum(255);
		rescaler->SetInput(img);
		itk::ImageFileWriter<CharImageType>::Pointer writer = itk::ImageFileWriter<CharImageType>::New();
		writer->SetInput(rescaler->GetOutput());
		stringstream grayFname;
		grayFname << DebugDirectory.string() <<"/GRAY" <<setw(4) << setfill('0') << SequenceNumber <<"_"<< beta_inv << "_" << SectionName << ".png";
		writer->SetFileName(grayFname.str());
		writer->Update();
	}

	return true;	
}


 
class StackAlign {
public:
	string brain, label;
	unsigned int ListCount, UsableCount;
	fs::path OutputDirectory, ListFilename, TransformFilename;
	vector<Section*> SectionList;
	vector<double> InitRotationList;
	vector<Links*> LinksList;
	double GlobalParameters[3];
	Section *startSection;
	
	StackAlign(string& brain, string& label);
	~StackAlign();
	bool ReadList(fs::path&,fs::path& );
	void Aligner(string&);
	bool IsGoodForStackAlignment(const unsigned int label);
	
	void ComputeAlignmentParameters(const int gx, const int gy);
	RGBPixelType getLargestClusterCenter(vector<RGBPixelType>& colVal) ;
	PixelType getScalingFactor(vector<RGBPixelType>& colVal);
	//double register2dMultiScaledWithTransformedImage(Section*curr, Section *next, TransformType::Pointer& finaltransform) ;
	double register2dMultiScaledWithTransformedImage(	ImageType::Pointer& fimg, double frot, 
																ImageType::Pointer& mimg, double mrot, 
																TransformType::Pointer& finaltransform,
																const int gx, const int gy);
	void WriteOutputImages( const unsigned int, const unsigned int);
	void WriteLinkParameters(string& recFileName, string& XformFileName);
	void clean();
	void WriteDiffImages( );

};

StackAlign::StackAlign(string& b, string& lbl) {
	brain = b;
	label = lbl;
	ListCount = 0;
	UsableCount = 0;
	GlobalParameters[0] = 0;
	GlobalParameters[1] = 0;
	GlobalParameters[2] = 0;
	ListFilename = fs::path("");
}

StackAlign::~StackAlign() {
}

bool StackAlign::ReadList(fs::path& listFName, fs::path& optdir) {
	
	ListFilename = fs::path(listFName);
	OutputDirectory = optdir;
	
	if (fs::exists(OutputDirectory) == true) {
		fs::remove_all(OutputDirectory);
	}
	fs::create_directory(OutputDirectory);

	
	if (debugON == true) {
		DebugDirectory = optdir;
		DebugDirectory /= "DEBUG";
		if (fs::exists(DebugDirectory) == true) {
			fs::remove_all(DebugDirectory);
		}
		fs::create_directory(DebugDirectory);
		cout << "Creating debug directory: " << DebugDirectory << endl;
	}
	
	ifstream fid(ListFilename.string().c_str(),ios::binary);
	if (fid.good() == false) {
		cout << "List file " << ListFilename.string() << " cannot be opened " << endl;
		return false;
	}
	DEBUGPRINT("Parsing: " << ListFilename.string() )
	
	while(fid.good() == true) {
		string line;
		ListCount++;
		getline(fid,line);
		DEBUGPRINT("Line: " << line)
		if (line.length() < 4) {
			continue;
		}
		if (line[0] == '#') {
			continue;
		}

		//cout << "Reading image file: " << line << endl;
		size_t n1 = line.find_last_of(":");
		double rot = 0;
		fs::path temp;
		if ( n1 == string::npos ) {
			temp = fs::path(line);
		}
		else {
			temp = fs::path(line.substr(0,n1));
			size_t n2 = line.length();
			istringstream(line.substr(n1+1,n2-n1)) >> rot;
		}
		if (fs::exists(temp) == true) {
			Section  *sec = new Section();
			sec->SectionName = temp.filename().string();
			size_t q = sec->SectionName.find_last_of(".");
			if ( q != string::npos) {	
				sec->SectionName = sec->SectionName.substr(0,q);
			}
			sec->SectionPath = temp;
			sec->SequenceNumber = UsableCount++;
			sec->Usable = true;
			if (debugON == true) { sec->Show(); }
			SectionList.push_back(sec);
			InitRotationList.push_back(rot);
			DEBUGPRINT( "Pre Rotation: " << rot );
			//cin.get();
		}
		else {
			cout << "File:" << temp.string() <<" doesnot exist." << endl << "Press Ctrl+C to quit, enter to continue" << endl;
			cin.get();
		}
	}
	DEBUGPRINT("# sections to process: " << SectionList.size() )
	fid.close();
	return true;
}


void StackAlign::ComputeAlignmentParameters( const int gx, const int gy) {
	DEBUGPRINT (endl << "Computing Alignment Parameters on " << SectionList.size() <<  " sections."  << endl)
	
	TransformType::Pointer Tranx = TransformType::New();
	Tranx->SetIdentity();
	
	
	double beta_inv = 25;
	ImageType::Pointer A, B;
	unsigned int start = 0;
	
	while (SectionList[start]->getPrepImage(beta_inv, A) == false) {
		SectionList[start]->Usable = false;
		start++;
		if (start > SectionList.size()) {
			cout << "All sections are not readable, please check " << endl;
			return;
		}
	}
	startSection = SectionList[start];
	SectionList[start]->Tx->SetParametersByValue(Tranx->GetParameters());
	SectionList[start]->Usable = true;
	SectionList[start]->prev = NULL;
	SectionList[start]->next = NULL;
	int lasti = start; 
	for (unsigned int i=start+1; i<SectionList.size(); i++) {		
		if (SectionList[i]->getPrepImage(beta_inv, B) == false) {
			SectionList[i]->Usable = false;
			continue;
		}
		SectionList[i]->prev = SectionList[lasti];
		SectionList[lasti]->next = SectionList[i];
		SectionList[i]->next = NULL;	
		cout << brain << "(" << label <<")    T"<<setw(3)<<setfill( '0' )<< SectionList[lasti]->SequenceNumber <<
			 " - T"<<setw(3)<<setfill( '0' )<< SectionList[i]->SequenceNumber <<" : "  ;
		
		double arot = InitRotationList[lasti], brot = InitRotationList[i];
		
		TransformType::Pointer Tx = TransformType::New();
		double error = register2dMultiScaledWithTransformedImage(A, arot, B, brot, Tx,	gx, gy);
		double angleDeg = 180.0*Tx->GetParameters()[0]/vnl_math::pi;
		DEBUGPRINT( angleDeg <<"\t" << error << "  " << Tx->GetParameters() )
		if (error >= 0.05) {
			DEBUGPRINT("Finding inverse :")
			TransformType::Pointer Tx2 = TransformType::New();
			double error2 = register2dMultiScaledWithTransformedImage(B, brot, A, arot, Tx2, gx, gy);	
			double angleDeg2 = 180.0*Tx2->GetParameters()[0]/vnl_math::pi;
			DEBUGPRINT( angleDeg2 <<"\t" << error2 << "  " << Tx2->GetParameters() )
			if (error2 < (error+0.02)) {
				DEBUGPRINT("Inverse better then direct, substituting...  ")
				Tx->SetCenter(Tx2->GetCenter());
				Tx2->GetInverse(Tx);
				error = error2;
				angleDeg = -1.0*angleDeg2;
				DEBUGPRINT( "T"<<setw(3)<<setfill( '0' )<< SectionList[lasti]->SequenceNumber <<
					 " - T"<<setw(3)<<setfill( '0' )<< SectionList[i]->SequenceNumber <<" : " << 
					 angleDeg <<"\t" << error << "  " << Tx->GetParameters() )
			}
			else {
				double beta_invList[4] = {5.0, 50.0, 100.0};
				double errorList, angleDegList;
				TransformType::Pointer TxList = TransformType::New();

				ImageType::PointType Aorig = A->GetOrigin();
				ImageType::PointType Borig = B->GetOrigin();

				for (int j=0; j<3; ++j) {
					DEBUGPRINT(  "Trying option with beta_inv " << beta_invList[j] )
					if ( (SectionList[start]->getPrepImage( beta_invList[j], A) )
							&& (SectionList[start]->getPrepImage( beta_invList[j], B) ) == false) {
						break;
					}
					A->SetOrigin(Aorig); SectionList[lasti]->origin = Aorig;
					B->SetOrigin(Borig); SectionList[i]->origin = Borig;
					errorList = register2dMultiScaledWithTransformedImage(A, arot, B, brot, Tx,	gx, gy);
					if (errorList < (error+0.02)) {
						DEBUGPRINT( "List better than direct, substituting...  " )
						Tx->SetParameters(TxList->GetParameters());
						error = errorList;
						angleDeg = angleDegList;
						DEBUGPRINT( "T"<<setw(3)<<setfill( '0' )<< SectionList[lasti]->SequenceNumber <<
							 " - T"<<setw(3)<<setfill( '0' )<< SectionList[i]->SequenceNumber <<" : " << 
							 angleDeg <<"\t" << error << "  " << Tx->GetParameters() << endl )
					}	
				}
			}
		}
		Links *lnk = new Links(SectionList[lasti],SectionList[i],Tx,error);
		LinksList.push_back(lnk);
		Tranx->Compose(Tx);
		SectionList[i]->Tx->SetParametersByValue(Tranx->GetParameters());
		lasti = i;
		A = B;
	}
}


void StackAlign::WriteLinkParameters(string& recFileName, string& XformFileName) {
	
	//write all LINKS again
	fs::path recPath = OutputDirectory;
	recPath /= recFileName;
	fs::path XformFname = OutputDirectory;
	XformFname /= XformFileName;
	
	ofstream ofid(recPath.string().c_str(),ios::binary);

	if (ofid.good() ==  false) {
		cout << "Parameter file " << recPath << " cannot be opened" << endl;
		return;
	}
	
	time_t rawtime;
	struct tm * timeinfo;
	
	time (&rawtime);
	timeinfo = localtime (&rawtime);
	cout << "Writing paremters ... "  ;
	
	ofid << "#Listfile:" << ListFilename.string() << endl;
	//ofid << "#ComposeDate:" << asctime(timeinfo) ;
	//ofid << "#GlobalParameters:"<<GlobalParameters[0]<<","<<GlobalParameters[1]<<","<<GlobalParameters[2]<<endl;
	
	unsigned int secCount = 0;
	for (unsigned int i=0; i<SectionList.size(); ++i) {
		
		if (SectionList[i]->Usable == true) {
			secCount++;
			ofid << "1,"<< SectionList[i]->SequenceNumber<<","	<< "G" << 
			"," << SectionList[i]->SectionName << 
			",Ox=" <<	SectionList[i]->origin[0] <<
			",Oy=" <<	SectionList[i]->origin[1] <<
			",Dx=" <<	SectionList[i]->direction[0] <<
			",Dy=" <<	SectionList[i]->direction[1] <<
			",Sx=" <<	SectionList[i]->size[0] <<
			",Sy=" <<	SectionList[i]->size[1] <<
			",Hf=0,Vf=0,Tf=0," << 
			//SectionList[i]->SectionPath.parent_path().string().c_str() << endl;
			SectionList[i]->SectionPath.string().c_str() << endl;
		}
		else {
			ofid << "0,"<< SectionList[i]->SequenceNumber<<","	<< "G" << 
			"," << SectionList[i]->SectionName << 
			",Ox=0,Oy=0,Dx=0,Dy=0,Sx=0,Sy=0,Hf=0,Vf=0,Tf=0," <<
			//SectionList[i]->SectionPath.parent_path().string().c_str() << endl;
			SectionList[i]->SectionPath.string().c_str() << endl;
		}
	}
	
	cout << secCount << " paramters written" << endl;
	
	cout << "Writing Links " ;
	int linkCount = 0;
	//sec#1,sec#2,rot,cx,cy,tx,ty,err
	for (unsigned int i=0; i<LinksList.size(); ++i) {
		linkCount++;
		Links *lnk = LinksList[i];
		TransformType::ParametersType param = lnk->Tx12->GetParameters();
		ofid << "$" << lnk->sec1->SequenceNumber << ","
		<< lnk->sec2->SequenceNumber << "," << param[0] << ","
		<< param[1] << ","<< param[2] << ","
		<< param[3] << ","<< param[4] << ","
		<< lnk->error << endl;
	}
	cout << linkCount << " links written" << endl;
	ofid.close();

	//cin.get();
	//Creating the Xform files
	ofstream xfid(XformFname.string().c_str());
	for ( vector<Section*>::iterator it = SectionList.begin(); it != SectionList.end(); ++it) {
		if ((*it)->Usable == false) {
			continue;
		}
		
		TransformType::ParametersType param = (*it)->Tx->GetParameters();
	
		xfid << (*it)->SectionName << "," 
		<< (*it)->SectionPath.parent_path().string() << ","
		<< (*it)->origin[0] << ","
		<< (*it)->origin[1] << ","
		<< (*it)->direction[0] << ","
		<< (*it)->direction[1] << ","
		<< (*it)->size[0]<< ","
		<< (*it)->size[1]<< ","
		<< param[0] << "," 
		<< param[1] << "," 
		<< param[2] << "," 
		<< param[3] << "," 
		<< param[4] << ","
		<< (int)(*it)->hflip << ","
		<< (int)(*it)->vflip << ","
		<< (int)(*it)->spProc << ","
		<< endl;	
	}
	
	xfid.close();
}


void StackAlign::WriteOutputImages( unsigned int gx ,  unsigned int gy ) {
	
	vector<Section*>::iterator it;
	cout << "Creating output images in " << OutputDirectory << endl;

	Section *curr = startSection;
	if (curr == NULL) {
		return;
	}

	while ( curr != NULL) {

	
		stringstream ss;
		ss << setw(4)<<setfill('0')<< curr->SequenceNumber<<"_";
	
		fs::path ofname = OutputDirectory;
		ofname /= ss.str();
		ofname += curr->SectionName;
		ofname += string(".png");
		//cout << "Writing " << ofname << endl;
		if (fs::exists(ofname) == true) {
			fs::remove(ofname);
		}
		
		//continue;
		
		if (curr->Usable == false) {
			curr = curr->next;
			continue;
		}
		TransformType::ParametersType param = curr->Tx->GetParameters();
		DEBUGPRINT( "Parameters: " << curr->SectionName << "," 
		<< curr->origin[0] << ","
		<< curr->origin[1] << ","
		<< curr->direction[0] << ","
		<< curr->direction[1] << ","
		<< curr->size[0]<< ","
		<< curr->size[1]<< ","
		<< param[0] << "," 
		<< param[1] << "," 
		<< param[2] << "," 
		<< param[3] << "," 
		<< param[4] << ","
		<< (int)curr->hflip << ","
		<< (int)curr->vflip << ","
		<< (int)curr->spProc << ","
		)
		
		itk::ImageFileReader<RGBImageType>::Pointer reader = itk::ImageFileReader<RGBImageType>::New();
		reader->SetFileName(curr->SectionPath.string());
		RGBImageType::Pointer imRGB = reader->GetOutput();
		imRGB->Update();
		imRGB->SetOrigin(curr->origin);
		double spacing[2]  = {1.0, 1.0};
		imRGB->SetSpacing(spacing);
		typedef itk::Matrix< double, 2, 2 > MatrixType;
		MatrixType Mi;
		Mi(0,0) = curr->direction[0];
		Mi(0,1) = curr->direction[1];
		Mi(1,0) = -1.0*curr->direction[1];
		Mi(1,1) = curr->direction[0];
		imRGB->SetDirection(Mi);
		itk::ImageRegionIterator< RGBImageType > it(imRGB,imRGB->GetBufferedRegion());
		
		//remove the cropping artifacts
		typedef itk::Image<unsigned char,2> MaskType;
		MaskType::Pointer mask = MaskType::New();
		mask->SetRegions(imRGB->GetBufferedRegion());
		mask->Allocate();
		mask->FillBuffer(0);
		itk::ImageRegionIterator< itk::Image<unsigned char,2> > itmask(mask,mask->GetBufferedRegion());

		vector<RGBPixelType> colVal;
		colVal.reserve(curr->size[0]*curr->size[1]);
		int mcnt = 0;
		RGBPixelType maxVal; maxVal.Fill(0);
		for (it.GoToBegin(), itmask.GoToBegin(); !it.IsAtEnd(); ++it, ++itmask) {
			RGBPixelType value = it.Get();
			if ((value[0]==0.0) && (value[1]==0.0) && (value[2]==0.0)) {
				mcnt++;
				itmask.Set(255);
			}
			else if ((value[0]==255.0) && (value[1]==255.0) && (value[2]==255.0)) {
				mcnt++;
				itmask.Set(255);
			}
			else {
				colVal.push_back(value);
				for (int i=0; i<3;++i) {
					maxVal[i] = (maxVal[i]>value[i]) ? maxVal[i] : value[i];
				}
			}
		}
		
		DEBUGPRINT( "Pixels count in mask: " << mcnt << " out of "  << curr->size[0]*curr->size[1] );
		DEBUGPRINT( "Max val " << maxVal )
		
		RGBPixelType DefPixelVal = getLargestClusterCenter(colVal);	
		DEBUGPRINT( "Default pixel " << DefPixelVal )
		
		for (it.GoToBegin(), itmask.GoToBegin(); !it.IsAtEnd(); ++it, ++itmask) {
			if (itmask.Get() == 255) {
				it.Set(DefPixelVal);
			}
		}	
		
		PixelType meanLum = getScalingFactor(colVal); 
		bool normRequired = (meanLum <= 1.1) ? false : true;

		typedef itk::ResampleImageFilter<RGBImageType, RGBImageType>  ResampleFilterType;
		ResampleFilterType::Pointer resample = ResampleFilterType::New();
		resample->SetTransform( curr->Tx );

		resample->SetInput( imRGB );
		DEBUGPRINT( "Size :" << imRGB->GetBufferedRegion().GetSize() )
		
		//cout << "Setting Resampling params...." << endl;
		ImageType::SizeType sz;
		sz[0] = gx;
		sz[1] = gy;
		//double origin[2], o1[2];
		double origin[2];
		origin[0] = -0.5*(double)gx; //OFFSET HERE
		origin[1] = -0.5*(double)gy;
		MatrixType M;
		M.SetIdentity();
		resample->SetOutputDirection(M);
		resample->SetSize( sz );
		resample->SetOutputOrigin( origin );
		resample->SetOutputSpacing( spacing );
		//RGBPixelType rgb; rgb.Fill(127);
		resample->SetDefaultPixelValue( DefPixelVal );
		RGBImageType::Pointer imopt16 = resample->GetOutput();
		//cout << "Resampling ...."<< endl;
		imopt16->Update();	               		
		//cout << "Resampled ...."<< endl;
		typedef itk::Image<itk::RGBPixel<unsigned char> , 2> ImageType2;
		ImageType2::Pointer imopt8 = ImageType2::New();
		imopt8->SetRegions(imopt16->GetBufferedRegion());
		imopt8->Allocate();
		
		itk::ImageRegionIterator<RGBImageType> it1(imopt16,imopt16->GetBufferedRegion());
		itk::ImageRegionIterator<ImageType2> it2(imopt8,imopt8->GetBufferedRegion());
			  
		for( it1.GoToBegin(), it2.GoToBegin(); !it1.IsAtEnd(); ++it1, ++it2) {
			itk::RGBPixel<unsigned char> d;	  
			for (int i=0; i<3; ++i) {
				unsigned short val = it1.Get()[i];
				if (normRequired == true) {
					val = val * 25.0/meanLum ; 
					//val = 200 + sqrt(val-200);
				}			
				d[i] = static_cast<unsigned char>(vnl_math_min(255,val));
			}
			it2.Set(d);
		}
		//cout << "8 bit conversion ...";		
		itk::ImageFileWriter<ImageType2>::Pointer writer = itk::ImageFileWriter<ImageType2>::New();
		writer->SetInput(imopt8);
		writer->SetFileName(ofname.string());
		writer->Update();
		//cout << "Written" << endl;
		curr = curr->next;
	}
	cout << "Writing complete " << endl;
}

PixelType StackAlign::getScalingFactor(vector<RGBPixelType>& colVal) {
	DEBUGPRINT("Computing scaling factor")
	bool is16bit = false;
	vector<PixelType> allPixels;
	allPixels.reserve(colVal.size()*3);
	DEBUGPRINT("Expected elemints" << colVal.size()*3 )
	for (unsigned int i=0; i<colVal.size(); ++i){
		for (int r=0; r<3; ++r) {
			if (colVal[i][r] > 255) {
				is16bit = true;
			}
			allPixels.push_back((PixelType)colVal[i][r]);
		}
	}
	DEBUGPRINT("Total elemints" << allPixels.size() )
	if (is16bit == false) {
		DEBUGPRINT( "Brightfield image -- No scaling");	
		return 1;
	}
	else {
		sort(allPixels.begin(), allPixels.end());
		size_t cutOff = allPixels.size()*9/10;
		PixelType cutOffVal = allPixels[cutOff];
		double meanLum = 0.0, count = 0.0;
		for(size_t i = 0; i < allPixels.size(); ++i) {
			if (allPixels[i] < cutOffVal) {
				meanLum += (double)allPixels[i];
				count += 1.0;
			}
		}
		meanLum /= count;
		DEBUGPRINT("Darkfield image with mean Lum "  << meanLum << " cutOff at " << cutOffVal << " Estimated from " << (int)count << " samples");	
		//meanLum *= 25.0;
		return static_cast<PixelType>(meanLum);
	}
}

RGBPixelType StackAlign::getLargestClusterCenter(vector<RGBPixelType>& colVal) {
	double meanCol[3] = {0,0,0};
	for (unsigned int i=0; i<colVal.size(); ++i){
		for (int r=0; r<3; ++r) {
			meanCol[r] += (double)colVal[i][r];
		}
	}
	
	//cout << "Totals ["  << meanCol[0] << "," << meanCol[1] << ","<< meanCol[2] <<   "]" << endl;
	for (int r=0; r<3; ++r) { 	meanCol[r] /= (double)colVal.size();}
	//cout << "Initial col ["  << meanCol[0] << "," << meanCol[1] << ","<< meanCol[2] <<   "]" << endl;
	
	unsigned int cutPerct[10] = {90,80,70,60,50,40,30,30,30,30};
	for (int iter = 0; iter < 10; ++iter)	{
		vector<double> dist(colVal.size(),0.0); 
		for (unsigned int i=0; i<colVal.size(); ++i){
			//dist[i] = 0;
			for (int r=0; r<3; ++r) { dist[i] += vnl_math_abs( meanCol[r] -  (double)colVal[i][r]);	}
		}
		sort(dist.begin(), dist.end());
		size_t cutPt = colVal.size()*cutPerct[iter]/100;
		double cutDist = dist[cutPt];
		//cout << "CutPt " << cutPt << "  Cutdist " << cutDist << endl;
		
		double meanColNew[3] = {0.0,0.0,0.0}, count = 0.0;
		for (unsigned int i=0; i<colVal.size(); ++i){
			double d = 0;
			for (int r=0; r<3; ++r) { 	d += vnl_math_abs( meanCol[r] -  (double)colVal[i][r]);	}
			if (d < cutDist) {
				for (int r=0; r<3; ++r) { meanColNew[r] += (double)colVal[i][r]; }
				count++;	
			}
		}
		for (int r=0; r<3; ++r) { meanCol[r] = meanColNew[r]/count;	}
		//cout << "Iter " << iter  << "  col ["  << meanCol[0] << "," << meanCol[1] << ","<< meanCol[2] << "]" << endl;
	}
	RGBPixelType ret; 
	for (int r=0; r<3; ++r) { ret[r] = (PixelType)meanCol[r];}
	return ret;
}


double StackAlign::register2dMultiScaledWithTransformedImage(	ImageType::Pointer& fimg, double frot, 
																ImageType::Pointer& mimg, double mrot, 
																TransformType::Pointer& finaltransform,
																const int gx, const int gy) {
	//ImageType::Pointer fimg = curr->img;
	//ImageType::Pointer mimg = next->img;
	//cout << " Fixed " << fimg << endl << 	"Moving" << mimg<< endl;
	//cout << " HERE...0"<< endl; 
	TransformType::Pointer Txf = TransformType::New();  
	//cout << " HERE...1"<< endl;
	Txf->SetIdentity();
	
	TransformType::Pointer Txm = TransformType::New();
	Txm->SetIdentity();
	
	ImageType::SizeType sz; 	sz[0] = gx; 	sz[1] = gy;
	ImageType::PointType origin;
	origin[0] = -1.0*(double)sz[0]/2.0; origin[1] = -1.0*(double)sz[1]/2.0;
	double spacing[2]  = {1.0, 1.0};
	//cout << " HERE...2"<< endl;
	typedef itk::ResampleImageFilter< ImageType, ImageType >    ResampleFilterType;
	ResampleFilterType::Pointer resample1 = ResampleFilterType::New();
	resample1->SetTransform( Txf );
	resample1->SetInput(fimg );
	resample1->SetSize( sz );
	resample1->SetOutputOrigin( origin );
	resample1->SetOutputSpacing( spacing );
	resample1->SetOutputDirection( fimg->GetDirection() );
	resample1->SetDefaultPixelValue( 0 );
	ImageType::Pointer fimg2 = resample1->GetOutput();
	fimg2->Update();	
	//cout << " HERE...3"<< endl;
	ResampleFilterType::Pointer resample2 = ResampleFilterType::New();
	resample2->SetTransform( Txm );
	resample2->SetInput(mimg );
	resample2->SetSize( sz );
	resample2->SetOutputOrigin( origin );
	resample2->SetOutputSpacing( spacing );
	resample2->SetOutputDirection( mimg->GetDirection() );
	resample2->SetDefaultPixelValue( 0 );
	ImageType::Pointer mimg2 = resample2->GetOutput();
	mimg2->Update();
	
	typedef itk::RegularStepGradientDescentOptimizer       OptimizerType;
	typedef itk::MeanSquaresImageToImageMetric<ImageType,ImageType >    MetricType;
	//typedef itk::NearestNeighborInterpolateImageFunction<ImageType, double  >    InterpolatorType; //SEE IF NEARESTNBR IMPROVES
	typedef itk::LinearInterpolateImageFunction<ImageType, double  >    InterpolatorType; //SEE IF NEARESTNBR IMPROVES
	//typedef itk::ImageRegistrationMethod< ImageType, ImageType >    RegistrationType;
	typedef itk::RecursiveMultiResolutionPyramidImageFilter<ImageType,ImageType  >    ImagePyramidType;
	typedef itk::MultiResolutionImageRegistrationMethod< ImageType,ImageType >    RegistrationType;


	MetricType::Pointer         metric        = MetricType::New();
	OptimizerType::Pointer      optimizer     = OptimizerType::New();
	InterpolatorType::Pointer   interpolator  = InterpolatorType::New();
	ImagePyramidType::Pointer fixedImagePyramid = ImagePyramidType::New();
	ImagePyramidType::Pointer movingImagePyramid = ImagePyramidType::New();
	RegistrationType::Pointer   registration  = RegistrationType::New();

	//cout << " HERE...4"<< endl;
	registration->SetMetric(        metric        );
	registration->SetOptimizer(     optimizer     );
	registration->SetFixedImagePyramid( fixedImagePyramid );
	registration->SetMovingImagePyramid( movingImagePyramid );
	registration->SetInterpolator(  interpolator  );
	registration->SetNumberOfLevels( 5 );

	finaltransform->SetIdentity();
	registration->SetTransform( finaltransform );
	TransformType::Pointer inittx = TransformType::New();
	inittx->SetIdentity();
	inittx->SetRotation( (frot - mrot)*vnl_math::pi/180.0 );
	//TransformType::InputVectorType initTrans;
	//initTrans[3] = (double)((rand()%200) - 100);
	//initTrans[4] = (double)((rand()%200) - 100);
	//cout << "Randomized paramters "	<< initTrans << endl;
	//inittx->SetTranslation(initTrans);
	
	registration->SetFixedImage(   fimg2   );
	registration->SetMovingImage(   mimg2   );
	registration->SetFixedImageRegion( fimg2->GetBufferedRegion() );
	registration->SetInitialTransformParameters( inittx->GetParameters() );
	cout << "Initial: " << inittx->GetParameters() << " ";
	
	typedef OptimizerType::ScalesType       OptimizerScalesType;
	OptimizerScalesType optimizerScales( finaltransform->GetNumberOfParameters() );
	const double translationScale = 0.01;
	const double rotationScale = 100.0 ;

	optimizerScales[0] = rotationScale;
	optimizerScales[1] = translationScale;
	optimizerScales[2] = translationScale;
	optimizerScales[3] = translationScale;
	optimizerScales[4] = translationScale;

	optimizer->SetScales( optimizerScales );

	optimizer->SetMaximumStepLength( 5.0    );
	optimizer->SetMinimumStepLength( 0.001 ); //TODO
	optimizer->SetNumberOfIterations( 500 );
	//cout << " HERE...5"<< endl;
	try	    {
	    registration->Update();
    }
	catch( itk::ExceptionObject & err )	{
		cerr << "ExceptionObject caught !" << endl;
		cerr << err << endl;
		return 1000.0;
    }
	//cout << " HERE...6"<< endl;
    //cout << "Optimizer stop condition: " << registration->GetOptimizer()->GetStopConditionDescription()  << endl;
 
	finaltransform->SetParameters(registration->GetLastTransformParameters());
	cout << "Final: " << finaltransform->GetParameters() ;
	//TransformType::ParametersType finaltransform_invpara = (dynamic_cast<TransformType *>(finaltransform->GetInverseTransform().GetPointer()))->GetParameters();
	//cout << "Final inverse Param:" << finaltransform_invpara << endl;
	cout << "Minimum value : " << optimizer->GetValue() << endl;
	
	if (debugON == true) {
		debugXformCount++;
		stringstream cntstr; cntstr << setw(4) << setfill('0') << debugXformCount << "-" << setw(4) << setfill('0') << debugXformCount+1 ;
		string str_f, str_m, str_mtx, str_diff;
		str_f = str_m = str_mtx = str_diff = DebugDirectory.string();
		//str_f.append("/FIXED_").append(cntstr.str()).append("_");
		//str_m.append("/MOVING_").append(cntstr.str()).append("_");
		//str_mtx.append("/MOVING_TRANSFORMED_").append(cntstr.str()).append("_");
		str_diff.append("/DIFF_").append(cntstr.str()).append("_");
		//WriteImage(str_f.c_str(), fimg2,gx,gy);
		//WriteImage(str_m.c_str(), mimg2,gx,gy);
		//WriteImage(str_mtx.c_str(), mimg2,finaltransform,gx,gy);
		WriteDiffImage(str_diff.c_str(), fimg2, mimg2,finaltransform,gx,gy);
		//cout << " Stopping for debug (Press any key to continue)" << endl;
		//cin.get();
	}
	
	return optimizer->GetValue();
}


void StackAlign::WriteDiffImages( ) {
	
	fs::path DiffImageDirectory = OutputDirectory;
	DiffImageDirectory /= "DIFF";
	
	if (fs::exists(DiffImageDirectory) == true) {
		fs::remove_all(DiffImageDirectory);
	}
	fs::create_directory(DiffImageDirectory);
	
	cout << "Computing DIFF images in " << DiffImageDirectory << endl;

	Section *curr = startSection;
	if (curr == NULL) {
		return;
	}
	typedef itk::Image<itk::RGBPixel<unsigned char> , 2> ImageType2;
	
	int last_seq_no = -1;
	ImageType2::Pointer last_imopt8;
	while ( curr != NULL) {

		stringstream ss;
		ss << setw(4)<<setfill('0')<< curr->SequenceNumber<<"_";
	
		fs::path ofname = OutputDirectory;
		ofname /= ss.str();
		ofname += curr->SectionName;
		ofname += string(".png");
		
		//if (fs::exists(ofname) == false) {
		//	continue;
		//}
		
		
		itk::ImageFileReader<ImageType2>::Pointer reader = itk::ImageFileReader<ImageType2>::New();
		reader->SetFileName(ofname.string());
		ImageType2::Pointer imopt8 = reader->GetOutput();
		try {
			imopt8->Update();
		}
		catch (itk::ExceptionObject e) {
			cout << "Error in reading" << ofname << endl << e.GetDescription() << endl;
			continue;
		}
		
		if (last_seq_no > 0) {
			/*ImageType2::Pointer diff = ImageType2::New();
			diff->SetRegions(imopt8->GetBufferedRegion());
			diff->Allocate();

			itk::ImageRegionIterator<ImageType2> it1( diff,diff->GetBufferedRegion() );
			itk::ImageRegionIterator<ImageType2> it2( imopt8,imopt8->GetBufferedRegion() );
			itk::ImageRegionIterator<ImageType2> it3( last_imopt8,last_imopt8->GetBufferedRegion() );
			for (it1.GoToBegin(),it2.GoToBegin(),it3.GoToBegin(); !it1.IsAtEnd(); ++it1, ++it2, ++it3) {
				ImageType2::PixelType p1;
				ImageType2::PixelType p2 = it2.Get();
				ImageType2::PixelType p3 = it3.Get();
				for (int i = 0; i<3; ++i) {
					p1[i] = 127 + p2[i]/2 - p3[i]/2;
				}
				it1.Set(p1);
			}
			*/
			itk::CheckerBoardImageFilter<ImageType2>::Pointer checkerbd =  itk::CheckerBoardImageFilter<ImageType2>::New();
			checkerbd->SetInput1(imopt8);
			checkerbd->SetInput2(last_imopt8);
			itk::CheckerBoardImageFilter<ImageType2>::PatternArrayType pattern;
			pattern.Fill(12);
			checkerbd->SetCheckerPattern(pattern);			
			
			stringstream dd;
			dd << setw(4)<<setfill('0')<< last_seq_no <<"_"<< setw(4)<<setfill('0') << curr->SequenceNumber;			
			
			fs::path dfname = DiffImageDirectory;
			dfname /= dd.str();
			//dfname += curr->SectionName;
			dfname += string(".png");
			//cout << "Writing " << dfname << endl;
			if (fs::exists(dfname) == true) {
				fs::remove(dfname);
			}

			
			itk::ImageFileWriter<ImageType2>::Pointer writer = itk::ImageFileWriter<ImageType2>::New();
			//writer->SetInput(diff);
			writer->SetInput(checkerbd->GetOutput());
			writer->SetFileName(dfname.string());
			writer->Update();
		}
		last_imopt8 = imopt8;
		last_seq_no = curr->SequenceNumber;
		curr = curr->next;
	}	
	cout << "Writing complete" << endl;
}




int main(int argc, const char **argv) {
	
	cout << endl << endl;
	if (argc < 2) {
		cout << "Usage : " << argv[0] << " List_file [ options ] " << endl 
			<< "Options: " << endl << "\t -brain (required)" << endl
								   << "\t -label (required)" << endl		
								   << "\t -output ( default: ./Output ) " << endl	
								   << "\t -beta_inv (range: 5-100, default: 25)"<< endl
								   << "\t -gx (ex: 900, default: 1000)"<< endl 
								   << "\t -gy (ex: 750, default: 1000)"<< endl
									;
		return EXIT_FAILURE;
	}		
	
	double beta_inv = 25.0;
	unsigned int gx = 1000;
	unsigned int gy = 1000;
	int i = 2;
	debugON = false;
	bool hasOutputpath = false;
	fs::path outputpath;
	string brain(""), label("");
		
	while (i < argc) {
		//cout << "arg# " << i << " out of " << argc << endl;
		//cout << "token " << argv[i] << endl;
		string token(argv[i]), value("");
		i++;
		if (i < argc) {
			//cout << "value " << argv[i] << endl;		
			value.assign(argv[i]);
			i++;	
		}
		
		if (token.compare("-beta_inv") == 0) {
			beta_inv = atof(value.c_str());
			cout << "beta inv: " << beta_inv << endl;
		}
		else if (token.compare("-gx") == 0) {
			gx = atoi(value.c_str());
			cout << "gx: " << gx << endl;
		}
		else if (token.compare("-gy") == 0) {
			gy = atoi(value.c_str());
			cout << "gy: " << gy << endl;
		}		
		else if (token.compare("-output") == 0) {
			outputpath = fs::path(value);
			hasOutputpath = true;
			cout << "output: " << outputpath.string() << endl;
		}		
		else if (token.compare("-brain") == 0) {
			brain = value;
			cout << "Brain: " << brain << endl;
		}		
		else if (token.compare("-label") == 0) {
			label = value;
			cout << "Label: " << label << endl;
		}		

		else if (token.compare("-debug") == 0) {
			debugON = true;
			cout << endl << "RUNNING IN DEBUG MODE " << endl << endl;
			
		}
		else {
			cout << "Token "<< token <<" cannot be identified" << endl;
			return EXIT_FAILURE;
		}	
		//cout << "OK" << endl;	
	}
	
	DEBUGPRINT("Completed Parsing input parameters")
	//cin.get();
	
	time_t starttime;
	time(&starttime); 
	//cin.get();
	string recFileName(""), XformFileName("");
	
	if ((brain.length() == 0) || (label.length() == 0)) {
		cout << "Parameter(s) -brain or -label missing from the parameter list." << endl << "Brain: " << brain << endl << "Label: " << label << endl;
		return EXIT_FAILURE;
	}
	recFileName = brain + string("_") + label + string(".txt");
	XformFileName = brain + string("_") + label + string("_XForm.txt");
	fs::path listFName = fs::path(argv[1]);
	//listFName = system_complete(listFName);
	DEBUGPRINT("Listfile : " << listFName  )
	DEBUGPRINT("Parameter FileName : " << recFileName  )
	DEBUGPRINT("Transform FileName : " << XformFileName  )
	
	
	if (hasOutputpath == false) {
		outputpath = fs::current_path();
		outputpath /= "Output";
	}
	
	//cout << "Creating new output directory: " << outputpath << endl;
	
	
	StackAlign *stack = new StackAlign(brain, label);  
	if (stack->ReadList(listFName, outputpath) == false) {
		cout << "Cannot read list file" << endl;
		return EXIT_FAILURE;
	}
	DEBUGPRINT("Parsed list file : " << listFName  )
	
	
	stack->ComputeAlignmentParameters( gx, gy);
	stack->WriteLinkParameters(recFileName, XformFileName);
	stack->WriteOutputImages( gx, gy );
	stack->WriteDiffImages();
	delete stack;
	
	time_t endtime; 
	time(&endtime);
	cout << " Complete in " <<difftime(endtime, starttime)/60.0 << " mins"<< endl;
	return EXIT_SUCCESS;		
}
