/* This version reads a parameter file and applys transform to an entire stack.
For this it reads each line from XForm file, applies transforms and saves images as pngs, 
then calls ImageMagick or FFMPEG to convert images to gif or flv */

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
#include "itkLinearInterpolateImageFunction.h"
#include "itkNearestNeighborExtrapolateImageFunction.h"
#include <itkRawImageIO.h>

#include <sys/time.h>
#include <boost/filesystem.hpp>
#include <unistd.h>
#include <sys/stat.h>

#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/warning.h>
#include <cppconn/metadata.h>
#include <cppconn/prepared_statement.h>
#include <cppconn/resultset.h>
#include <cppconn/resultset_metadata.h>
#include <cppconn/statement.h>
#include "mysql_driver.h"
#include "mysql_connection.h"

using namespace std;
namespace fs = boost::filesystem;


const unsigned int Dimension = 2;
typedef unsigned short PixelType;
typedef itk::RGBPixel<unsigned short> RGBPixelType;
typedef itk::Image<RGBPixelType,Dimension> ImageType;
typedef itk::CenteredRigid2DTransform<double> TransformType;

bool getDirectoryFromDatabase(string& SectionName,string& directory, sql::Statement *& stmt) {
	// string url("tcp://mitramba1.cshl.edu:3306");
	// const string user("registration");
	// const string password("admin");
	// const string database("MBAStorageDB");
	// sql::Driver * driver = sql::mysql::get_driver_instance();
	// sql::Connection *con = driver->connect(url, user, password);
	// sql::Statement *stmt = con->createStatement();
	// try {		
		// stmt->execute("use " + database);
	// }
	// catch (sql::SQLException &e) {
		// cout << "SQLException "<< e.what();
		// cout << " MySQL error code: " << e.getErrorCode();
		// cout << ", SQLState: " << e.getSQLState() << " )" << endl;
		// return false;
	// }
	stringstream query;
	query << "SELECT path FROM Navigator_section WHERE filename= \"" << SectionName << "\";" ;
	try {
		cout << "Launching query: " << query.str() << endl;
		std::auto_ptr< sql::ResultSet > res(stmt->executeQuery(query.str().c_str()));
		if (res->rowsCount() == 0) {
			cout << "Section information not in the database" << endl;
			return false;
		}
		else  {
			res->next(); 
			string db_directory(res->getString(1));
			string MDrive;
			size_t kk = 16;
			for (size_t nn = 16; nn < 18; ++nn) {
				char c = db_directory[nn];
				if ((c>='0') && (c<='9')) {
					MDrive.push_back(c);
				}
				c = directory[nn];
				if ((c>='0') && (c<='9')) {
					kk = nn+1;
				}
			}
			cout << "Parameter file path:" << directory << endl;
			cout << "Query returned path:" << db_directory << endl;
			cout << "new path:" << directory.substr(0,16) + MDrive + directory.substr(kk) << endl;
			directory = directory.substr(0,16) + MDrive + directory.substr(kk);
			cout << "M drive name:" << MDrive << endl;
			//just replace the correct MDrive path
			if (directory[directory.length()-1] == '/') {
				directory.erase(directory.length()-1);	
			}
			
			//cin.get();
		}
	}
	catch (sql::SQLException &e) {
		cout << "SQLException "<< e.what();
		cout << " MySQL error code: " << e.getErrorCode();
		cout << ", SQLState: " << e.getSQLState() << " )" << endl;
		return false;
	}
	
	return true; 
}

class Xform{

private:
	int  hflip,vflip,sproc;
	double R, D[2], O[2], A[2], T[2],S[2];
	double spacing[2];
	itk::Matrix<double,2,2> direction;
	size_t outWidth, outHeight;
	
	string 	BrainName,
			SectionName,
			directory,
			tempfolder,
			ifname,
			ofname;
	fs::path iPath, oPath; 
	TransformType::Pointer Tx;
	itk::Size<2> iSize;
	ImageType::PointType origin;

public:	
	Xform(const size_t, const size_t);
	bool parseParamline(string& line, string& tempfolder, int, sql::Statement *& stmt);
	bool adjustParameters();
	PixelType getMode(vector<PixelType>& vect );
	bool execute();
	RGBPixelType getDefPixelVal(ImageType::Pointer&);
	RGBPixelType getLargestClusterCenter(vector<RGBPixelType>&colVal);
};


Xform::Xform(const size_t gx, const size_t gy) {
	R = 0.0; 
	for (int i=0;i<2;++i){
		S[i] = O[i] = A[i] = T[i] = D[i] = 0.0;
	}
	outWidth = gx;
	outHeight = gy;
	
	iSize[0] = 0;
	iSize[1] = 0;
	
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


bool Xform::parseParamline(string& line, string& tempfolder, int numIndex, sql::Statement *& stmt) {
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
	<<"Hflip = " << hflip << " Vflip = " << vflip << " Special = " << sproc << endl;

	if (tempfolder.find("UNKNOWN") != string::npos) {
		BrainName.assign("UNKNOWN");
		//PMD789&788-F29-2012.09.23-04.30.55_PMD788_3_0087
		n1 = SectionName.find_first_of("_");
		if (n1 != string::npos) {
			n2 = SectionName.find_first_of("_",n1+1);
			if (n2 != string::npos) {
				BrainName = SectionName.substr(n1+1,n2-n1-1);
			}
		}
		tempfolder = BrainName;
		fs::path tempfolderPath(tempfolder);
		tempfolderPath = fs::system_complete(tempfolderPath);
		tempfolder = tempfolderPath.string();
		if (fs::exists(tempfolderPath) == false) {
			fs::create_directory(tempfolderPath);
		}
	}
	
	if (getDirectoryFromDatabase(SectionName,directory, stmt) == false) {
		return false;
	}
	
	ifname = SectionName + ".tif";
	iPath = directory;
	iPath /= ifname;

	stringstream ndx;
	ndx << setw(4)<<setfill('0')<< numIndex << "_"<< SectionName << ".png";
	ofname = ndx.str();	
	oPath = tempfolder;
	oPath /= ofname;
	

	cout << "Input " << iPath.string() << endl << "Output " << oPath.string() << endl;

	origin[0] = O[0];
	origin[1] = O[1];
	
	direction(0,0) = D[0];
	direction(0,1) = D[1];
	direction(1,0) = -1.0*D[1];
	direction(1,1) = D[0];
	
	spacing[0] = 1.0;
	spacing[1] = 1.0;

	return true;
}

PixelType Xform::getMode(vector<PixelType>& vect ) {
	vector<PixelType>::iterator it = vect.begin();
	PixelType maxVal = (*it); 
	PixelType minVal = (*it);
	while(it != vect.end()) {
		if((*it) > maxVal) { maxVal = (*it);}
		if((*it) < minVal) { minVal = (*it);}
		++it;
	}
	int range = (int)(maxVal - minVal + 1);
	cout << "Def hist range: high-" << maxVal << " low-" << minVal << endl;
	vector<int> hist(range,0);
	it = vect.begin();
	while(it != vect.end()) {
		int ndx = (*it)-minVal;
		if ((ndx < 0) || (ndx >= range)) {
			cout << "Out of range intensity" << endl;
		}
		else {
			hist[ndx]++;
		}
		++it;
	}
	int mode = 1; maxVal = hist[1];
	for (unsigned int h = 1; h < hist.size(); ++h) {
		//cout << "Def hist " << h << "[" << hist[h] << "]" << endl;
		if (hist[h] > maxVal) {
			maxVal = hist[h];
			mode = h;
		}
	}
	return (PixelType)(mode + minVal);
}


RGBPixelType Xform::getLargestClusterCenter(vector<RGBPixelType>&colVal) {
	unsigned int meanCol[3] = {0,0,0};
	for (unsigned int i=0; i<colVal.size(); ++i){
		for (int r=0; r<3; ++r) {
			meanCol[r] += (unsigned int)colVal[i][r];
		}
	}
	
	//cout << "Totals ["  << meanCol[0] << "," << meanCol[1] << ","<< meanCol[2] <<   "]" << endl;
	for (int r=0; r<3; ++r) { 	meanCol[r] /= colVal.size();}
	//cout << "Initial col ["  << meanCol[0] << "," << meanCol[1] << ","<< meanCol[2] <<   "]" << endl;
	
	unsigned int cutPerct[10] = {90,80,70,60,50,40,30,30,30,30};
	for (int iter = 0; iter < 10; ++iter)	{
		vector<unsigned int> dist(colVal.size(),0); 
		for (unsigned int i=0; i<colVal.size(); ++i){
			//dist[i] = 0;
			for (int r=0; r<3; ++r) { dist[i] += (unsigned int)vnl_math_abs( (int)meanCol[r] -  (int)colVal[i][r]);	}
		}
		sort(dist.begin(), dist.end());
		size_t cutPt = colVal.size()*cutPerct[iter]/100;
		unsigned int cutDist = dist[cutPt];
		//cout << "CutPt " << cutPt << "  Cutdist " << cutDist << endl;
		
		unsigned int meanColNew[3] = {0,0,0}, count = 0;
		for (unsigned int i=0; i<colVal.size(); ++i){
			unsigned int d = 0;
			for (int r=0; r<3; ++r) { 	d += (unsigned int)vnl_math_abs( (int)meanCol[r] -  (int)colVal[i][r]);	}
			if (d < cutDist) {
				for (int r=0; r<3; ++r) { meanColNew[r] += (unsigned int)colVal[i][r]; }
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


RGBPixelType Xform::getDefPixelVal(ImageType::Pointer& img) {
	RGBPixelType defBG;
	for (int col=0; col < 3; ++col) {
		vector<PixelType> valList(10000);
		size_t h = 30;
		for (size_t y=0; y<img->GetBufferedRegion().GetSize(1);++y) {
			for (size_t x=0; x<h;++x) {
				itk::Index<2> ndx = {{x,y}};
				PixelType m = img->GetPixel(ndx)[col];
				if ((m != 0) | (m != 255)) {
					valList.push_back(m);
				}
				ndx[0] = img->GetBufferedRegion().GetSize(0)-1-x;
				m = img->GetPixel(ndx)[col];
				if ((m != 0) | (m != 255)) {
					valList.push_back(m);
				}
			}
		}
		for (size_t x=0; x<img->GetBufferedRegion().GetSize(0);++x) {
			for (size_t y=0; y<h;++y) {
				itk::Index<2> ndx = {{x,y}};
				PixelType m = img->GetPixel(ndx)[col];
				if ((m != 0) | (m != 255)) {
					valList.push_back(m);
				}
				ndx[1] = img->GetBufferedRegion().GetSize(1)-1-y;
				m = img->GetPixel(ndx)[col];
				if ((m != 0) | (m != 255)) {
					valList.push_back(m);
				}
			}
		}
		
		//sort(valList.begin(),valList.end());
		//size_t midpt = valList.size()/2;
		//defBG[col] = valList.at(midpt);
		defBG[col] = getMode(valList);
	}
	cout << "Default pixel value :" << defBG << endl;
	return defBG;
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
	
		
	itk::ImageFileReader<ImageType>::Pointer reader = itk::ImageFileReader<ImageType>::New();
	reader->SetFileName(iPath.string());
	try {
		reader->Update();
	}
	catch (itk::ExceptionObject e) {
		e.Print(cout);
		return false;
	}
	
	ImageType::Pointer img = reader->GetOutput();
	//check image size and adjust parameters here
	iSize = img->GetBufferedRegion().GetSize();
	if (adjustParameters() == false) {
		cout << "Parameters not matching the image" << endl;
		return false;;	
	}
	
	img->SetOrigin(origin);
	img->SetSpacing(spacing);
	img->SetDirection(direction);
	itk::ImageRegionIterator<ImageType> it1(img,img->GetBufferedRegion());
	
	//create a mask of the same size and fill it with 0
	typedef itk::Image<unsigned char,2> MaskType;
	MaskType::Pointer mask = MaskType::New();
	mask->SetRegions(img->GetBufferedRegion());
	mask->Allocate();
	mask->FillBuffer(0);
	itk::ImageRegionIterator< itk::Image<unsigned char,2> > itmask(mask,mask->GetBufferedRegion());
	
	int mcnt = 0;
	for (it1.GoToBegin(), itmask.GoToBegin(); !it1.IsAtEnd(); ++it1, ++itmask) {
		RGBPixelType value = it1.Get();
		if ((value[0]==0) && (value[1]==0) && (value[2]==0)) {
			mcnt++;
			itmask.Set(255);
		}
		if ((value[0]==255) && (value[1]==255) && (value[2]==255)) {
			mcnt++;
			itmask.Set(255);
		}
	}
	cout << "Pixels in mask: " << mcnt << " out of "  << iSize[0]*iSize[1] << endl;
	
	//determine if brightfield or darkfield
	vector<PixelType> defVal;
	defVal.reserve(iSize[0]*iSize[1]); 	
	vector<RGBPixelType> colVal;
	colVal.reserve(iSize[0]*iSize[1]);
	
	for (it1.GoToBegin(), itmask.GoToBegin(); !it1.IsAtEnd(); ++it1, ++itmask) {
		if (itmask.Get() == 255) {
			continue;
		}
		RGBPixelType value = it1.Get();
		colVal.push_back(value);
		//cout << value << endl;
		if ((value[0]>value[1]) && (value[0]>value[2])) {
			defVal.push_back(value[0]);
		}
		else if ((value[1]>value[0]) && (value[1]>value[2])) {
			defVal.push_back(value[1]);
		}
		else {
			defVal.push_back(value[2]);
		}
	}
	
	cout << "Foregd pixels " << defVal.size() << endl;
	sort(defVal.begin(),defVal.end());

	size_t medianPos = defVal.size()/2;
	int lowCutPercentile = 25;
	int highCutPercentile = 75;
	int lowlowCutPercentile = 10;
	int highhighCutPercentile = 90;

	size_t lenLow = defVal.size()*lowCutPercentile/100;
	size_t lenHigh = defVal.size()*highCutPercentile/100;
	size_t lenLowLow = defVal.size()*lowlowCutPercentile/100;
	size_t lenHighHigh = defVal.size()*highhighCutPercentile/100;

	
	PixelType medianVal = defVal[medianPos];
	PixelType lowLimit = defVal[lenLow];
	PixelType highLimit = defVal[lenHigh];
	PixelType lowlowLimit = defVal[lenLowLow];
	PixelType highhighLimit = defVal[lenHighHigh];
	
	cout << "Input image very low value " << lowlowLimit << endl;
	cout << "Input image low value " << lowLimit << endl;
	cout << "Input image medial value " << medianVal << endl;
	cout << "Input image high value " << highLimit << endl;
	cout << "Input image very high value " << highhighLimit << endl;
	
	
	
	RGBPixelType DefPixelVal = getLargestClusterCenter(colVal);	
	bool isBrightField = false;
	PixelType scale1 , scale2;
	
	if (medianVal > 100) {
		cout << "Brightfield image" <<endl;
		isBrightField = true;
		scale1 = 100;	//25%
		scale2 = 240; 	//75%
	}
	else {
		cout << "Darkfield image" << endl;
		scale1 = 10;    //25%
		scale2 = 100;  //75%
	}
	cout << "Default pixel " << DefPixelVal << endl;

	for (it1.GoToBegin(), itmask.GoToBegin(); !it1.IsAtEnd(); ++it1, ++itmask) {
		if (itmask.Get() == 255) {
			it1.Set(DefPixelVal);
		}
	}	
	//cin.get();	
	
	//special processing
	if (sproc == 1) {
		//rescale intensities by 16
		PixelType BITSHIFT = 8;
		
		for (it1.GoToBegin(); !it1.IsAtEnd(); ++it1) {
			RGBPixelType val = it1.Get();
			val[0] /= BITSHIFT;
			val[1] /= BITSHIFT;
			val[2] /= BITSHIFT;
			it1.Set(val);
		}
	}
	
	//typedef itk::BSplineInterpolateImageFunction<ImageType,double,double> InterpolatorType;
	typedef itk::LinearInterpolateImageFunction<ImageType,double> InterpolatorType;
	InterpolatorType::Pointer resampler = InterpolatorType::New();
	//typedef itk::NearestNeighborExtrapolateImageFunction<ImageType, double> ExtrapolatorType;
	//ExtrapolatorType::Pointer extrapolator = ExtrapolatorType::New();
	typedef itk::ResampleImageFilter< ImageType, ImageType >    ResampleFilterType;
	ResampleFilterType::Pointer resample = ResampleFilterType::New();
	resample->SetTransform( Tx );
	resample->SetInput( img );
	resample->SetInterpolator(resampler);
	//resample->SetExtrapolator(extrapolator);
	//resample->SetDefaultPixelValue(getDefPixelVal(img));
	resample->SetDefaultPixelValue(DefPixelVal);
	ImageType::SizeType sz;
	sz[0] = outWidth;
	sz[1] = outHeight;
	//outWidth, outHeight
	double start[2], startXformed[2];
	start[0] = -1.0*double(outWidth)/2.0 ;
	start[1] = -1.0*double(outHeight)/2.0 ;
	startXformed[0] = D[0]*start[0] - D[1]*start[1];
	startXformed[1] = D[1]*start[0] + D[0]*start[1]; 
	//ImageType::PointType orig[2], o1[2];
	// o1[0] = -0.5*(double)outWidth; //OFFSET HERE
	// o1[1] = -0.5*(double)outHeight;
	// orig[0] = direction[0]*o1[0] - direction[1]*o1[1];
	// orig[1] = direction[1]*o1[0] + direction[0]*o1[1]; 
	

	resample->SetSize( sz );
	resample->SetOutputOrigin( startXformed );
	resample->SetOutputSpacing( spacing );

	ImageType::Pointer imopt16 = resample->GetOutput();
	imopt16->Update();
	
	typedef itk::RGBPixel<unsigned char> OutputRGBPixelType;
	typedef itk::Image<OutputRGBPixelType, Dimension> OutputRGBImageType;

	OutputRGBImageType::Pointer imopt8 = OutputRGBImageType::New();
	imopt8->SetRegions(sz);
	imopt8->Allocate();
	
	itk::ImageRegionIterator<ImageType> it(imopt16,imopt16->GetBufferedRegion().GetSize());
	itk::ImageRegionIterator<OutputRGBImageType> oit(imopt8,imopt8->GetBufferedRegion().GetSize());
	
	
	for (it.GoToBegin(), oit.GoToBegin(); !it.IsAtEnd(); ++it, ++oit) {
		RGBPixelType value = it.Get();
		OutputRGBPixelType ovalue;
		for (int r=0; r<3;++r) {
			//value[r] = (value[r] - lowLimit)*((scale2 - scale1)/(highLimit - lowLimit)) + scale1;
			value[r] = (value[r] > 255) ? 255 : value[r];
			ovalue[r] = static_cast<unsigned char>(value[r]);
		}
		oit.Set(ovalue);
	}
	
	/*vector<PixelType> normVal(sz[0]*sz[1]); 
	for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
		RGBPixelType value = it.Get();
		if (isBrightField == true) {
			if ((value[0] == 255) && (value[1] == 255) && (value[2] == 255)) {
				continue;
			}
		}
		else {
			if ((value[0] == 0) && (value[1] == 0) && (value[2] == 0)) {
				continue;
			}		
		}
		normVal.push_back((value[0]+value[1]+value[2])/3);
	}
	sort(normVal.begin(),normVal.end());
	
	unsigned int TotLum = 0, count = 0;
	for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
		RGBPixelType value = it.Get();
		unsigned int lum = (unsigned int)(value[0]+value[1]+value[2])/3;
		if ((lum >= lowLimit) && (lum <= highLimit)) {
			TotLum += lum;
			count++;
		}
	}
	unsigned int aveLum = TotLum/count;
	PixelType intRange = highLimit - lowLimit;
	cout << "Normalizing factors : " << endl <<
		"\thighLimit: " << highLimit << endl <<
		"\tlowLimit: " << lowLimit << endl <<
		"\trange: " << intRange <<	endl <<
		"\tavgLum: " << aveLum <<	endl ;	
	
	OutputRGBPixelType ZERO;
	ZERO[0] = 0;
	ZERO[1] = 0;
	ZERO[2] = 0;
	PixelType baseInt = lowLimit + 30;
	if (isBrightField == false) {
		//darkfield
		cout << "ImageType: Darkfield" << endl;
		for (oit.GoToBegin(),it.GoToBegin(); !it.IsAtEnd(); ++it, ++oit) {
			RGBPixelType value = it.Get();
			PixelType lum = (value[0]+value[1]+value[2])/3;
			if (lum < baseInt) {
				oit.Set(ZERO);
			}
			else {
				OutputRGBPixelType d;
				for (int i = 0; i < 3; ++i) {
					value[i] = (value[i] > baseInt) ? (value[i] - baseInt) : 0;
					value[i] = value[i] * 25 / aveLum ;
					value[i] = (value[i] > 255) ? 255 : value[i];
					d[i] = static_cast<unsigned char>(value[i]);
				}
				oit.Set(d);
			}
		}
	}
	else {
		//brightfield
		cout << "ImageType: Brightfield" << endl;
		for (oit.GoToBegin(),it.GoToBegin(); !it.IsAtEnd(); ++it, ++oit) {
			RGBPixelType value = it.Get();
			OutputRGBPixelType d;
			for (int i = 0; i < 3; ++i) {
				d[i] = static_cast<unsigned char>(value[i]);
			}
			oit.Set(d);
		}
	}
	*/
	itk::ImageFileWriter<OutputRGBImageType>::Pointer writer = itk::ImageFileWriter<OutputRGBImageType>::New();
	writer->SetFileName(oPath.string());
	writer->SetInput(imopt8);
	writer->Update();
	cout << "Output written in " << oPath.string() << endl;

	return true;
}




int main( int argc, char * argv[] ) {
	
	if ((argc < 2) || (argc > 3)) {
		cout << "Usage: XFormFile.txt [OutputFolder]" <<	endl;
		cout << "Supplied " << argc << " arguements" << endl; 
		return EXIT_FAILURE;
	}

	size_t gx = 24000/32;
	size_t gy = 18000/32;
	
	string XFormFile(argv[1]), param;
	ifstream fid(XFormFile.c_str(),ios::binary);
	
	if (fid.good() == false) {
		cout << "Cannot open " << XFormFile << endl;
		return EXIT_FAILURE;
	}
	
	string tempfolder("UNKNOWN");
	if (argc == 3) {
		tempfolder.assign(argv[2]);
	}
	else {
		//parse the filename to get tempfoldername
		size_t n1 = XFormFile.find_first_of("_");
		if (n1 != string::npos) {
			tempfolder = XFormFile.substr(0,n1);
			tempfolder.append("_XFORMED_IMAGES");
		}
	}
	fs::path tempfolderPath(tempfolder);
	tempfolderPath = fs::system_complete(tempfolderPath);
	tempfolder = tempfolderPath.string();
	if (fs::exists(tempfolderPath) == true) {
		fs::remove_all(tempfolderPath);
	}
	fs::create_directory(tempfolderPath);
	
	string url("tcp://mitramba1.cshl.edu:3306");
	const string user("registration");
	const string password("admin");
	const string database("MBAStorageDB");
	sql::Driver * driver = sql::mysql::get_driver_instance();
	sql::Connection *con = driver->connect(url, user, password);
	sql::Statement *stmt = con->createStatement();
	try {		
		stmt->execute("use " + database);
	}
	catch (sql::SQLException &e) {
		cout << "SQLException "<< e.what();
		cout << " MySQL error code: " << e.getErrorCode();
		cout << ", SQLState: " << e.getSQLState() << " )" << endl;
		return false;
	}
	
	Xform xf(gx,gy);
//string param("PMD789&788-F26-2011.09.09-19.16.00_PMD788_1_0076,/nfs/data/main/M8/mba_converted_imaging_data/PMD789&788/PMD788_JP2,-328.08,-169.625,0.969053,-0.246851,661,516,-0.0372065,-28.194,-17.092,-72.439,93.4285,0,0,0,");
	int count = 0, success = 0;
	while (fid.good() == true) {
		getline(fid,param);
		if (param.length() < 10)
			continue;
		if (xf.parseParamline(param,tempfolder, count++, stmt) == false) {
			cout << "skipping lines " << param << endl;
			continue;			
		}
		
		
		

		if (xf.execute() == true) {
			success++;
		}
		else {
			cout  << endl<< endl <<"Stopping because image missing" << endl;
			ofstream logid("LOG_FILE_OF_UNSUCCESSFUL_RUNS.txt",ios::binary| ios::app); 
			logid << param << endl; 
			logid.close();
			fs::remove_all(tempfolderPath);
			fid.close();
			cout << "Transformation of " << XFormFile << " is UNSUCCESSFUL, " << success << " out of " << count << " processed. Exiting..."  << endl;
			delete con;
			delete stmt;
			return EXIT_FAILURE;
			//cin.get();
		}
		//
	}
	fid.close();
	cout << "Successfully converted " << success << " out of " << count << endl;
	delete con;
	delete stmt;
	return EXIT_SUCCESS;
}


