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
#include <sys/time.h>
#include <boost/filesystem.hpp>
#include <unistd.h>
#include <sys/stat.h>
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


#include "itkArray.h"

//#include "StackAlign.h"
//#include "StackAlign.h"

using namespace std;
namespace fs = boost::filesystem;

const unsigned int Dimension = 2;
typedef  float PixelType;
bool debugON;
fs::path DebugDirectory;

typedef itk::Image<unsigned char,Dimension> CharImageType;
typedef itk::Image<PixelType,Dimension> ImageType;
typedef itk::RGBPixel<PixelType> RGBPixelType;
typedef itk::Image<RGBPixelType, Dimension> RGBImageType;
typedef itk::CenteredRigid2DTransform<double> TransformType;

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
	try {
		imRGB->Update();
	}

	catch (itk::ExceptionObject e) {
		cout << "Error in reading" << this->SectionPath.string() << endl << e.GetDescription() << endl;
		return false;
	}
	
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
	

	cout << "R = " << mR << ", G = " << mG << ", B = " << mB << endl;
	
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
	
	//cout << "Data Size: " << dist.size() << " Midloc: " << midloc << endl;
	//cout << "Median: " << medDist << endl;
	
	double fgmean = 0.0, fcount = 0.0;
	for ( vector<double>::iterator it5 = dist.begin(); it5 != dist.end(); ++it5) {
		if ((*it5) > medDist) {
			fgmean += vcl_log(*it5);
			fcount += 1.0;
		}
	}

	fgmean /= fcount;
	fgmean = vcl_exp(fgmean); 
	cout << "Fg mean: " << fgmean << endl;
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
	cout << "Data Size: " << m << " Cut 1: " << dist[cut1] <<  " Cut 2: " << dist[cut2] << endl;
	
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
	cout << "mX = " << orig[0] << ", mY = " << orig[1] << endl;
	
	if (debugON == true) {
		itk::RescaleIntensityImageFilter<ImageType,CharImageType>::Pointer rescaler = itk::RescaleIntensityImageFilter<ImageType,CharImageType>::New();
		rescaler->SetOutputMinimum(0);
		rescaler->SetOutputMaximum(255);
		rescaler->SetInput(img);
		itk::ImageFileWriter<CharImageType>::Pointer writer = itk::ImageFileWriter<CharImageType>::New();
		writer->SetInput(rescaler->GetOutput());
		stringstream grayFname;
		grayFname << DebugDirectory.string() <<"/GRAY_" << beta_inv << "_" << SectionName << ".png";
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
	bool ReadList(const char*);
	bool ReadList(const char*p, fs::path& optdir);
	void Aligner(string&);
	bool IsGoodForStackAlignment(const unsigned int label);
	
	void ComputeAlignmentParameters(const int gx, const int gy);
	//double register2dMultiScaledWithTransformedImage(Section*curr, Section *next, TransformType::Pointer& finaltransform) ;
	double register2dMultiScaledWithTransformedImage(	ImageType::Pointer& fimg, double frot, 
																ImageType::Pointer& mimg, double mrot, 
																TransformType::Pointer& finaltransform,
																const int gx, const int gy);
	void WriteOutputImages( const unsigned int, const unsigned int);
	void WriteLinkParameters(string& recFileName, string& XformFileName);
	void clean();

};

StackAlign::StackAlign(string& b, string& lbl) {
	brain = b;
	label = lbl;
	ListCount = 0;
	UsableCount = 0;
	GlobalParameters[0] = 0;
	GlobalParameters[1] = 0;
	GlobalParameters[2] = 0;
}

StackAlign::~StackAlign() {
}

bool StackAlign::ReadList(const char*p ) {

	fs::path listFName = fs::path(p);
	listFName = system_complete(listFName);
	
	fs::path optdir = listFName.parent_path();
	optdir /= "Output";
	if (fs::exists(optdir) == true) {
		fs::remove_all(optdir);
	}
	fs::create_directory(optdir);
	cout << "Creating output directory: " << optdir << endl;
	
	return (ReadList(p,optdir));

}

bool StackAlign::ReadList(const char*p, fs::path& optdir) {
	
	ListFilename = fs::path(p);
	ListFilename = system_complete(ListFilename);
	
	OutputDirectory = optdir;
	
	if (debugON == true) {
		DebugDirectory = ListFilename.parent_path();
		DebugDirectory /= "DEBUG";
		if (fs::exists(DebugDirectory) == true) {
			fs::remove_all(DebugDirectory);
		}
		fs::create_directory(DebugDirectory);
	}
	
	ifstream fid(ListFilename.string().c_str(),ios::binary);
	if (fid.good() == false) {
		return false;
	}
	
	while(fid.good() == true) {
		string line;
		ListCount++;
		getline(fid,line);
		if (line.length() < 4) {
			continue;
		}
		if (line[0] == '#') {
			continue;
		}

		cout << "Reading image file: " << line << endl;
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
			sec->Show();
			SectionList.push_back(sec);
			InitRotationList.push_back(rot);
			cout << "Pre Rotation: " << rot << endl;
			//cin.get();
		}
		else {
			cout << "...... image file deosnot exist." << endl << "Press Ctrl+C to quit, enter to continue" << endl;
			cin.get();
		}
	}
	fid.close();
	return true;
}


void StackAlign::ComputeAlignmentParameters( const int gx, const int gy) {
	cout << endl << "ComputeAlignmentParameters"<< endl;
	
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
		cout << brain << ":" << label <<" T"<<setw(3)<<setfill( '0' )<< SectionList[lasti]->SequenceNumber <<
			 " - T"<<setw(3)<<setfill( '0' )<< SectionList[i]->SequenceNumber <<" : "  ;
		
		double arot = InitRotationList[lasti], brot = InitRotationList[i];
		
		TransformType::Pointer Tx = TransformType::New();
		double error = register2dMultiScaledWithTransformedImage(A, arot, B, brot, Tx,	gx, gy);
		double angleDeg = 180.0*Tx->GetParameters()[0]/vnl_math::pi;
		cout << angleDeg <<"\t" << error << "  ";
		cout << Tx->GetParameters() << endl;
		if (error >= 0.05) {
			cout << "Finding inverse :";	
			TransformType::Pointer Tx2 = TransformType::New();
			double error2 = register2dMultiScaledWithTransformedImage(B, brot, A, arot, Tx2, gx, gy);	
			double angleDeg2 = 180.0*Tx2->GetParameters()[0]/vnl_math::pi;
			cout << angleDeg2 <<"\t" << error2 << "  ";
			cout << Tx2->GetParameters() << endl;
			if (error2 < (error+0.02)) {
				cout << "Inverse better then direct, substituting...  " << endl;
				Tx->SetCenter(Tx2->GetCenter());
				Tx2->GetInverse(Tx);
				error = error2;
				angleDeg = -1.0*angleDeg2;
				cout << "T"<<setw(3)<<setfill( '0' )<< SectionList[lasti]->SequenceNumber <<
					 " - T"<<setw(3)<<setfill( '0' )<< SectionList[i]->SequenceNumber <<" : "  ;
				cout << angleDeg <<"\t" << error << "  ";
				cout << Tx->GetParameters() << endl;				
			}
			else {
				double beta_invList[4] = {5.0, 50.0, 100.0};
				double errorList, angleDegList;
				TransformType::Pointer TxList = TransformType::New();

				ImageType::PointType Aorig = A->GetOrigin();
				ImageType::PointType Borig = B->GetOrigin();

				for (int j=0; j<3; ++j) {
					cout << "Trying option with beta_inv " << beta_invList[j] <<endl;
					if ( (SectionList[start]->getPrepImage( beta_invList[j], A) )
							&& (SectionList[start]->getPrepImage( beta_invList[j], B) ) == false) {
						break;
					}
					A->SetOrigin(Aorig); SectionList[lasti]->origin = Aorig;
					B->SetOrigin(Borig); SectionList[i]->origin = Borig;
					errorList = register2dMultiScaledWithTransformedImage(A, arot, B, brot, Tx,	gx, gy);
					if (errorList < (error+0.02)) {
						cout << "List better than direct, substituting...  " << endl;
						Tx->SetParameters(TxList->GetParameters());
						error = errorList;
						angleDeg = angleDegList;
						cout << "T"<<setw(3)<<setfill( '0' )<< SectionList[lasti]->SequenceNumber <<
							 " - T"<<setw(3)<<setfill( '0' )<< SectionList[i]->SequenceNumber <<" : "  ;
						cout << angleDeg <<"\t" << error << "  ";
						cout << Tx->GetParameters() << endl << endl;
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
			SectionList[i]->SectionPath.parent_path().string().c_str() << endl;
		}
		else {
			ofid << "0,"<< SectionList[i]->SequenceNumber<<","	<< "G" << 
			"," << SectionList[i]->SectionName << 
			",Ox=0,Oy=0,Dx=0,Dy=0,Sx=0,Sy=0,Hf=0,Vf=0,Tf=0," <<
			SectionList[i]->SectionPath.parent_path().string().c_str() << endl;
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
		cout << "Parameters: " << curr->SectionName << "," 
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
		<< endl;
		
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

		typedef itk::ResampleImageFilter<RGBImageType, RGBImageType>  ResampleFilterType;
		ResampleFilterType::Pointer resample = ResampleFilterType::New();
		resample->SetTransform( curr->Tx );

		resample->SetInput( imRGB );
		cout << "Size :" << imRGB->GetBufferedRegion().GetSize() << endl;
		
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
		RGBPixelType rgb; rgb.Fill(127);
		resample->SetDefaultPixelValue( rgb );
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
		
		unsigned int dave = 0, cnt = 0;
		float maxval = 0.0;	
		for( it1.GoToBegin(), it2.GoToBegin(); !it1.IsAtEnd(); ++it1, ++it2) {
			for (int i=0; i<3; ++i) {
				maxval = vnl_math_max(maxval,it1.Get()[i]);
				dave += it1.Get()[i]; 
				cnt++; 	
			}
		}
		  
		dave /= cnt;
		if (dave <= 25) {dave = 20;}
		if (dave >= 80) {dave = 80;}
		if (maxval <= 255.0) {
			dave = 25;
		}
		//cout << "Ave intensity : " << dave << " ...."<< endl;
			  
		for( it1.GoToBegin(), it2.GoToBegin(); !it1.IsAtEnd(); ++it1, ++it2) {
			itk::RGBPixel<unsigned char> d;	  
			for (int i=0; i<3; ++i) {
				unsigned short val = it1.Get()[i]*25/ dave;
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
		
		string str_f, str_m, str_mtx;
		str_f = str_m = str_mtx = DebugDirectory.string();
		str_f.append("/FIXED_");
		str_m.append("/MOVING_");
		str_mtx.append("/MOVING_TRANSFORMED");
		WriteImage(str_f.c_str(), fimg2,gx,gy);
		WriteImage(str_m.c_str(), mimg2,gx,gy);
		WriteImage(str_mtx.c_str(), mimg2,finaltransform,gx,gy);
		cout << " Stopping for debug (Press any key to continue)" << endl;
		cin.get();
	}
	
	return optimizer->GetValue();
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
	
	while (i < (argc-1)) {
		string token(argv[i++]);
		string value(argv[i++]);
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
			cout << endl << "RUNNING IN DEBUG MODE (PRESS ENTER)" << endl << endl;
			cin.get();
			i--;
		}
		else {
			cout << "Token "<< token <<" cannot be identified" << endl;
			return EXIT_FAILURE;
		}		
	}

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
	
	if (hasOutputpath == false) {
		fs::path listFName = fs::path(argv[1]);
		listFName = system_complete(listFName);
		outputpath = listFName.parent_path();
		outputpath /= "Output";
	}
	
	if (fs::exists(outputpath) == true) {
		fs::remove_all(outputpath);
	}
	fs::create_directory(outputpath);
	cout << "Creating new output directory: " << outputpath << endl;
	
	
	StackAlign *stack = new StackAlign(brain, label);  
	if (stack->ReadList(argv[1], outputpath) == false) {
		cout << "Cannot read list file" << endl;
		return EXIT_FAILURE;
	}
	//cin.get();
	
	stack->ComputeAlignmentParameters( gx, gy);
	stack->WriteLinkParameters(recFileName, XformFileName);
	stack->WriteOutputImages( gx, gy );
	delete stack;
	
	time_t endtime; 
	time(&endtime);
	cout << " Complete in " <<difftime(endtime, starttime)/60.0 << " mins"<< endl;

	return EXIT_SUCCESS;		
}
