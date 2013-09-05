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
	<< "ModeIndex: " << SequenceNumber <<
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

class StackRepair {
public:
	double GlobalParameters[3];
	fs::path OutputDirectoryOLD, OutputDirectoryNEW,
	ParameterFileOLD, ParameterFileNEW, TransformFilenameNEW;
	vector<Section*> SectionList;
	vector<double> InitRotationList;
	vector<Links*> LinksList;
	Section *startSection;
	string brain, label;
	bool parameter_file_found;
	unsigned int gx, gy;
	vector<unsigned int> ExcludedSections;
	vector< pair<unsigned int, unsigned int> > ExcludedLinks;
	
	StackRepair(const char*, const char*, unsigned int, unsigned int);
	~StackRepair();
	bool ReadList(const char*);
	bool ReadList(const char*p, fs::path& optdir); 
	void Aligner(string&);
	void ResolveLinks();
	bool ReadRecord();
	bool IsGoodForStackAlignment(const unsigned int label);
	double LinkSections(Section*sec1, Section *sec2, TransformType::Pointer& Tx);
	bool parseParam(string param, string& tag, double & value); 
	bool parseSectionLine(Section*sec,string& line);
	void parseLinkLine(Links* link, string& line);
	bool parseExclusions(string& exclu);
	
	void ComposeStack(const double* off);
	//void ComputeAlignmentParameters(int, const int gx, const int gy);
	//double register2dMultiScaledWithTransformedImage(Section*curr, Section *next, TransformType::Pointer& finaltransform) ;
	double register2dMultiScaledWithTransformedImage(	ImageType::Pointer& fimg, double frot, 
																ImageType::Pointer& mimg, double mrot, 
																TransformType::Pointer& finaltransform);
	//void WriteOutputImages(int, const unsigned int, const unsigned int);
	//void WriteLinkParameters();
	void WriteOutputImages();
	void WriteTransforms();
	void clean();

};

StackRepair::StackRepair(const char* oldDir, const char* newDir, unsigned int gx_, unsigned int gy_) {
	gx = gx_;
	gy = gy_;
	OutputDirectoryOLD = fs::path(oldDir);
	OutputDirectoryNEW = fs::path(newDir);
	//scan the old directory for Parameter file
	fs::directory_iterator end ;
	parameter_file_found = false;
	string fname;
	for( fs::directory_iterator iter(OutputDirectoryOLD) ; iter != end ; ++iter ) {
		if ( is_regular_file(*iter) == true) {
			string fname = iter->path().filename().string();
			size_t pos = fname.find(".txt");
			if (( pos != string::npos) && (((int)fname.length() - (int)pos) == 4)) {
				cout << "Txt file in " <<OutputDirectoryOLD.string() << " found " << fname << endl; 
				ParameterFileOLD = iter->path();
				size_t n1 = fname.find_first_of("_");
				size_t n2 = fname.find_last_of("_");
				if (( n1 == string::npos) || (n1 > pos) || (n1 != n2)) {
					continue;
				}
				cout << "Parameter file found " << fname << endl;
				brain = fname.substr(0,n1);
				label = fname.substr(n1+1,pos-n1-1);
				cout << "Brain: " << brain << "  Label:" << label << endl; 
				parameter_file_found = true;
				break;
			}
		}
	}
	
	if (parameter_file_found == false) {
		return;
	}
	string recFileName = brain + string("_") + label + string(".txt");
	string XformFileName = brain + string("_") + label + string("_XForm.txt");
 
	ParameterFileNEW = OutputDirectoryNEW;
	ParameterFileNEW /= recFileName;
	TransformFilenameNEW = OutputDirectoryNEW;
	TransformFilenameNEW /= XformFileName;
	if (debugON == true) {
		DebugDirectory = OutputDirectoryNEW;
		DebugDirectory /= "DEBUG";
		if (fs::exists(DebugDirectory) == true) {
			fs::remove_all(DebugDirectory);
		}
		fs::create_directory(DebugDirectory);
	}
	if (fs::exists(OutputDirectoryNEW) == true) {
		cout << "Detected output directory " << OutputDirectoryNEW.string() << " ... clearing contents." << endl;
		fs::remove_all(OutputDirectoryNEW);
	}
	fs::create_directory(OutputDirectoryNEW);
}

StackRepair::~StackRepair() {}

bool StackRepair::parseParam(string param, string& tag, double & value) {
	size_t n = param.find_first_of('=');
	if (n != string::npos) {
		tag = param.substr(0,n);
		istringstream(param.substr(n+1)) >> value;
		return true;
	}
	return false;
}


bool StackRepair::parseSectionLine(Section*sec,string& line) {
	//1,105,F,PMD789&788-F35-2011.09.10-15.56.46_PMD788_3_0105,Ox=-447.95,Oy=-233.457,Dx=0.956245,Dy=-0.292567,Sx=783,Sy=616,Hf=0,Vf=0,Tf=0,path
	//cout << line << endl;
	size_t n1 = 0, n2 = line.find_first_of(',');
	int state = 0;

	string tag; double value;
	while (n2 != string::npos) {
		switch(state) {
		case 0:
			istringstream(line.substr(n1,n2-n1)) >> sec->Usable; state++; break;		
		case 1:
			istringstream(line.substr(n1,n2-n1)) >> sec->SequenceNumber; state++; break;
		case 2: 
			state++; break;
		case 3:	
			sec->SectionName = line.substr(n1,n2-n1); state++; break;
		case 4:
			value = 0.0;
			if (parseParam(line.substr(n1,n2-n1), tag, value) == false) {
				break;
			}
			if (tag.compare("Ox")==0) {sec->origin[0] = value;}
			else if (tag.compare("Oy")==0) {sec->origin[1] = value;}
			else if (tag.compare("Dx")==0) {sec->direction[0] = value;}
			else if (tag.compare("Dy")==0) {sec->direction[1] = value;}
			else if (tag.compare("Sx")==0) {sec->size[0] = value;}
			else if (tag.compare("Sy")==0) {sec->size[1] = value;}
			else if (tag.compare("Hf")==0) {sec->hflip = value;}
			else if (tag.compare("Vf")==0) {sec->vflip = value;}
			else if (tag.compare("Tf")==0) {sec->spProc = value;}
		}
		n1=n2+1;
		n2 = line.find_first_of(',',n1);
	}
	string directory = line.substr(n1);
	sec->SectionPath = directory;
	sec->SectionPath /= sec->SectionName;
	sec->SectionPath += ".tif";
	//sec->img = NULL;
	return true;
}

void StackRepair::parseLinkLine(Links* link, string& line) {
	size_t n1 = 1,n2 = line.find_first_of(',',n1);
	unsigned int tempSec1 = 0, tempSec2 = 0;
	itk::Array<double> param(5);
	param.Fill(0.0);
	double error;
	int state = 0;
	while (n2 != string::npos) {
		//cout << "...State.."<<state<< endl;
		switch(state) {
		case 0:
			istringstream(line.substr(n1,n2-n1)) >> tempSec1; state++; break;
		case 1:
			istringstream(line.substr(n1,n2-n1)) >> tempSec2; state++; break;
		case 2:
			istringstream(line.substr(n1,n2-n1)) >> param[0]; state++; break;
		case 3:
			istringstream(line.substr(n1,n2-n1)) >> param[1]; state++; break;
		case 4:
			istringstream(line.substr(n1,n2-n1)) >> param[2]; state++; break;	
		case 5:
			istringstream(line.substr(n1,n2-n1)) >> param[3]; state++; break;
		case 6:
			istringstream(line.substr(n1,n2-n1)) >> param[4]; state++; break;	
		//case 7:
			//istringstream(line.substr(n1,n2-n1)) >> error; state++; break;	
		}
		n1 = n2+1;
		n2 = line.find_first_of(',',n1);
	}
	
	istringstream(line.substr(n1)) >> error;	
	
	//scan for the Sec pointers
	link->sec1 = NULL; link->sec2 = NULL;
	for (vector<Section*>::iterator it = SectionList.begin(); it != SectionList.end(); ++it) {
		if ((*it)->SequenceNumber == tempSec1) {
			link->sec1 = (*it);
			//cout << ".. LINK1 found.." <<tempSec1 ;
		}
		else if ((*it)->SequenceNumber == tempSec2) {
			link->sec2 = (*it);
			//cout << ".. LINK2 found.." <<tempSec2 ;
		}
	}
	link->Tx12 = TransformType::New();
	link->Tx12->SetParametersByValue(param);
	link->error = error;
	//cout <<endl<< "...Param " << param << " " << error << endl;
	//cin.get();
	//return true;
}


double StackRepair::LinkSections(Section*sec1, Section *sec2, TransformType::Pointer& Tx) {
	
	double beta_invList[4] = {5.0, 50.0, 100.0};
	ImageType::Pointer A, B;
	double minerror = 100.0;
	cout << brain << ":" << label <<" T"<<setw(3)<<setfill( '0' )<< sec1->SequenceNumber <<
	 " - T"<<setw(3)<<setfill( '0' )<< sec2->SequenceNumber <<" : "  ;

	for (int i=0; i<3; ++i) {
		if ( (sec1->getPrepImage(beta_invList[i], A) == true) && 
			(sec2->getPrepImage(beta_invList[i], B) == true) )	{
				
			double arot[3] = {-30, 0, 30};
			double brot = 0;
			for (int j=0; j<3; ++j) {
				TransformType::Pointer Tx1 = TransformType::New();
				double error1 = register2dMultiScaledWithTransformedImage(A, arot[j], B, brot, Tx1);
				double angleDeg = 180.0*Tx1->GetParameters()[0]/vnl_math::pi;
				
				if (error1 < minerror) {
					minerror = error1;
					Tx->SetParametersByValue(Tx1->GetParameters());
					//cout << angleDeg <<"\t" << error1 << "  ";
					//cout << Tx->GetParameters() << endl;					
				}
				
				TransformType::Pointer Tx2 = TransformType::New();
				double error2 = register2dMultiScaledWithTransformedImage(B, brot, A, arot[j], Tx2);	
				double angleDeg2 = 180.0*Tx2->GetParameters()[0]/vnl_math::pi;
				if (error2 < minerror) {
					minerror = error2;
					Tx->SetCenter(Tx2->GetCenter());
					Tx2->GetInverse(Tx);
					//cout << -1.0*angleDeg2 <<"\t" << error2 << "  ";
					//cout << Tx->GetParameters() << endl;
				}
			}
		}
	}
	
	cout << "**********************************************" << endl;
	cout << Tx->GetParameters() << endl;
	cout << "**********************************************" << endl;
	return minerror;
}

double StackRepair::register2dMultiScaledWithTransformedImage(	ImageType::Pointer& fimg, double frot, 
																ImageType::Pointer& mimg, double mrot, 
																TransformType::Pointer& finaltransform) {
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
	cout << "Initial:" << inittx->GetParameters() ;
	
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
	cout << "\t\tFinal : " << finaltransform->GetParameters();
	//TransformType::ParametersType finaltransform_invpara = (dynamic_cast<TransformType *>(finaltransform->GetInverseTransform().GetPointer()))->GetParameters();
	//cout << "Final inverse Param:" << finaltransform_invpara << endl;
	cout << "\t\tMinimum value : " << optimizer->GetValue() << endl;
	
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



void StackRepair::ComposeStack(const double* off) {
	//
	TransformType::Pointer Tz = TransformType::New();
	Tz->SetIdentity();
	TransformType::InputPointType zero, zerodash, zerosum, cgShift;
	double zeroCount = 0.0;
	zero.Fill(0.0);
	zerosum.Fill(0.0);
		
	Section *sec = startSection;
	while (sec != NULL) {
		bool linkFound = false;
		for (vector<Links*>::iterator it2 = LinksList.begin(); it2 != LinksList.end(); ++it2) {
			if ((*it2)->sec1 == NULL) 	{ 	continue; }
			if ((*it2)->sec2 == NULL) 	{	continue; }
			if (sec->next == NULL) 	{	break;	}
			if (((*it2)->sec1->SequenceNumber == sec->SequenceNumber)  && ((*it2)->sec2->SequenceNumber == sec->next->SequenceNumber)) {
				//cout << "Desired link found " << sec->SequenceNumber << " and " << sec->next->SequenceNumber << endl;
				Tz->Compose((*it2)->Tx12);
				linkFound = true;
				break;
			}
		}
		 
		if ((linkFound == false) && (sec->next != NULL)) {
			cout << "Link not found" << endl;
			continue;
		}
		
		zerodash = Tz->TransformPoint(zero);
		zerosum[0] += zerodash[0];
		zerosum[1] += zerodash[1];
		zeroCount += 1.0;
		//cout << zeroCount  <<") " << sec->SequenceNumber << " " << zerodash << endl;		
		sec = sec->next;
	}
	cgShift[0] = zerosum[0]/zeroCount;
	cgShift[1] = zerosum[1]/zeroCount;
	cout << "CG shift = " << cgShift << endl; 
	//cin.get();
	//cgShift[0] = 15.9095;
	//cgShift[1] = 44.3621;
	
	double Xoff = off[1]*cos(off[0]) - off[2]*sin(off[0]);
	double Yoff = off[1]*sin(off[0]) + off[2]*cos(off[0]);
	
	//all the links are present.
	TransformType::Pointer Tranx = TransformType::New();
	Tranx->SetIdentity();
	TransformType::ParametersType InitParam = Tranx->GetParameters();
	InitParam[0] = off[0];
	//InitParam[1] = 0.0;
	//InitParam[2] = 0.0;
	//InitParam[1] = -1.0*(double)startSection->size[0]/2.0;
	//InitParam[2] = -1.0*(double)startSection->size[1]/2.0;
	InitParam[1] = -1.0*(double)startSection->size[0]/2.0 - startSection->origin[0];
	InitParam[2] = -1.0*(double)startSection->size[1]/2.0 - startSection->origin[1];
	InitParam[3] = -1.0*Xoff - cgShift[0];
	InitParam[4] = -1.0*Yoff - cgShift[1];
	//InitParam[3] = -1.0*off[1] - cgShift[0];
	//InitParam[4] = -1.0*off[2] - cgShift[1];
	Tranx->SetParameters(InitParam);
	cout << "Start Param" << Tranx->GetParameters() << endl;
	//startSection->Show();
	//cin.get();
	
	Section *curr = startSection;
	while (curr != NULL) {
		curr->Tx->SetParametersByValue(Tranx->GetParameters());
		//cout << "Composing "<< curr->SequenceNumber <<  " " << curr->Tx->GetParameters() << endl;
		bool linkFound = false;
		for (vector<Links*>::iterator it2 = LinksList.begin(); it2 != LinksList.end(); ++it2) {
			if ((*it2)->sec1 == NULL) {	continue;	}
			if ((*it2)->sec2 == NULL) {	continue;	}
			if (curr->next == NULL) {	break;		}
			if (((*it2)->sec1->SequenceNumber == curr->SequenceNumber)  && ((*it2)->sec2->SequenceNumber == curr->next->SequenceNumber)) {
				//cout << "Desired link found " << curr->SequenceNumber << " and " << curr->next->SequenceNumber << endl;
				Tranx->Compose((*it2)->Tx12);
				linkFound = true;
				break;
			}
		}
		if ((linkFound == false) && (curr->next != NULL)) {
			cout << "Link not found" << endl;
			//cin.get();
		}
		curr = curr->next;
	}
}


void StackRepair::ResolveLinks() {
	
	vector < bool > SegOutLinkOK(SectionList.size(),false),
	SegInLinkOK(SectionList.size(),false);
	
	unsigned minMode = 1000, maxMode = 0, counter = 0;
	for (unsigned int i=0; i < SectionList.size(); ++i) {
		if (SectionList[i]->Usable == false) {
			continue;
		}
		if (SectionList[i]->SequenceNumber > maxMode) {
			maxMode = SectionList[i]->SequenceNumber;
		}
		if (SectionList[i]->SequenceNumber < minMode) {
			minMode = SectionList[i]->SequenceNumber;
		}
		counter++;
	}
	
	cout << "Max SeqIndex: " << maxMode << endl 
	<< "Min SeqIndex: " << minMode << endl
	<< "Total num Sections: " << SectionList.size() << "(" << counter << " usable)"<< endl;
	
	//for every node, find is all the links exists, if not then make one
	unsigned int prevNdx = 0, currNdx;
	while((fs::exists(SectionList[prevNdx]->SectionPath) == false) || 
			(SectionList[prevNdx]->Usable == false)){
		cout << "File does not exist or not usable: "<< SectionList[prevNdx]->SectionPath << endl;
		prevNdx++;
	}
	minMode = SectionList[prevNdx]->SequenceNumber;
	startSection = SectionList[prevNdx];
	
	for ( currNdx=prevNdx+1; currNdx < SectionList.size(); ++currNdx ) {
		if (SectionList[currNdx]->Usable == false) {
			continue;
		}
		if (fs::exists(SectionList[currNdx]->SectionPath) == false) {
			cout << "File does not exist: " << SectionList[currNdx]->SectionPath << endl;
			SectionList[currNdx]->Usable = false;
			continue;
		}
		unsigned int sec1Mode = SectionList[prevNdx]->SequenceNumber;
		unsigned int sec2Mode = SectionList[currNdx]->SequenceNumber;
		if (sec1Mode == sec2Mode) {
			continue;
		}
		bool linkFound = false;
		for (vector<Links*>::iterator it2 = LinksList.begin(); it2 != LinksList.end(); ++it2) {
			if ((*it2)->sec1 == NULL) {
				continue;
			}
			if ((*it2)->sec2 == NULL) {
				continue;
			}
			
			if (((*it2)->sec1->SequenceNumber == sec1Mode)  && ((*it2)->sec2->SequenceNumber == sec2Mode)) {
				//cout << "Desired link found " << sec1Mode << " and " << sec2Mode << endl;
				linkFound = true;
				break;
			}
		}
		if (linkFound == false) {
			cout << "Link needed between " << sec1Mode << " and " << sec2Mode << endl;
			Section *sec1 = SectionList[prevNdx];
			Section *sec2 = SectionList[currNdx];
			TransformType::Pointer Tx12 = TransformType::New();			
			double error = LinkSections(sec1, sec2, Tx12);
			cout << "Error " << error << endl << Tx12->GetParameters() << endl;
			Links *link = new Links(sec1,sec2,Tx12,error);
			LinksList.push_back(link);
		}
		SectionList[prevNdx]->next = SectionList[currNdx];
		SectionList[currNdx]->prev = SectionList[prevNdx];
		SectionList[currNdx]->next = NULL;
		prevNdx = currNdx;
	}

}



bool StackRepair::parseExclusions(string& exclu) {
//{4,5,3,2-3,3-4,3}
	bool brOpen = false, isLink = false;
	size_t n1 = 1, n2;
	unsigned int val1, val2;
	
	for (unsigned int i=0; i<exclu.length(); ++i) {
		switch (exclu[i]) {
		case '[':	
			brOpen = true; 
			n1 = i+1;
			break;	
		case ',':  
			n2 = i;
			if (isLink==true) {
				istringstream(exclu.substr(n1,n2-n1)) >> val2;
				cout << val1 << ",," << val2 << endl;
				pair<int, int> p(val1, val2);
				ExcludedLinks.push_back(p);				
			}
			else {
				istringstream(exclu.substr(n1,n2-n1)) >> val1;
				ExcludedSections.push_back(val1);
			}
			isLink = false;
			n1 = n2+1;
			break;	
		case '-':
			n2 = i;
			istringstream(exclu.substr(n1,n2-n1)) >> val1;
			isLink = true;
			n1 = n2+1;
			break;
		case ']' :
			n2 = i;
			if (isLink==true) {
				istringstream(exclu.substr(n1,n2-n1)) >> val2;
				cout << val1 << "," << val2 << endl;
				pair<int, int> p(val1, val2);
				ExcludedLinks.push_back(p);
			}
			else {
				istringstream(exclu.substr(n1,n2-n1)) >> val1;
				ExcludedSections.push_back(val1);
			}
			if (brOpen == true) {
				brOpen = false;
			}
			break;	
		}		
	}
	cout << "Excluded sections: ";
	for (unsigned int i=0;i<ExcludedSections.size();++i){
		cout << ExcludedSections[i] <<",";
	}
	cout << endl;
	cout << "Excluded links: ";
	for (unsigned int i=0;i<ExcludedLinks.size();++i){
		cout << ExcludedLinks[i].first <<"-" <<ExcludedLinks[i].second <<",";
	}
	cout << endl;
		
	return (brOpen==false);
}


bool StackRepair::ReadRecord() {
	ifstream fid(ParameterFileOLD.string().c_str(),ios::binary);
	if (fid.good() ==  false) {
		cout << "Paraneter file " << ParameterFileOLD.string() << " cannot be opened" << endl;
		return false;
	}
	cout << "Parameter file opened successfully" << endl;
	string line;
	while (fid.good() == true) {
		getline(fid,line);
		//cout << line << endl;
		if (line.length() < 1) {
			continue;
		}
		if (line[0] == '1') {
			//cout << "usable" << endl;
			Section *sec =  new Section();
			if (parseSectionLine(sec,line) == true) {
				//sec->Show();
				SectionList.push_back(sec);
			}
			else {
				delete sec;
			}
		}
		else if (line.find("#GlobalParameters") != string::npos) {
			//ofid << "#GlobalParameters:"<<GlobalParameters[0]<<","<<GlobalParameters[1]<<","<<GlobalParameters[2]<<endl
			size_t n1 = line.find_first_of(":") , n2 = line.find_first_of(",",n1);
			for (int m = 0; m<3; ++m){
				//cout << "n1: " << n1 << " n2: " << n2 << endl;
				if ((n1 != string::npos) && (n2 != string::npos)) {
					cout << line.substr(n1+1,n2-n1) << endl;
					istringstream(line.substr(n1+1,n2-n1)) >> GlobalParameters[m];
				}
				n1 = n2;
				n2 = line.find_first_of(",",n1+1);
				if (n2 == string::npos) {n2 = line.length()-1;} 
			}
			cout << "GlobalParameters in record : " << GlobalParameters[0]<<","<<GlobalParameters[1]<<","<<GlobalParameters[2]<<endl;
			//cout << "Verify parameters" << endl;
			//cin.get();
		}
	}
	fid.close();
	cout << "Section parsed: " << SectionList.size() << endl;
	
	//Supress excluded sections
	int exclCounter = 0;
	for(unsigned int j=0;j<ExcludedSections.size();++j) {
		cout << "Excl: " << ExcludedSections[j]  << endl;
		for (unsigned int i=0; i<SectionList.size();++i){
			if (ExcludedSections[j] == SectionList[i]->SequenceNumber) {
				SectionList[i]->Usable = false;
				exclCounter++;
			}
		}
	}
	cout << "Supressed " << exclCounter << " sections." << endl;
	
	cout << "Reading links" << endl;
	fid.open(ParameterFileOLD.string().c_str(),ios::binary);
	fid.seekg(ios_base::beg);
	while (fid.good() == true) {
		getline(fid,line);
		//cout << line << endl;
		if (line.length() < 1) {
			continue;
		}
		if (line[0] == '$') {
			//cout << "link" << endl;
			Links *link = new Links();
			parseLinkLine(link,line);
			if (( link->sec1 != NULL) && (link->sec2 != NULL)) {
				//link->Print();
				//check excluded links
				bool badlink = false;
				for(unsigned int j=0;j<ExcludedLinks.size();++j) {
					if (((link->sec1->SequenceNumber == ExcludedLinks[j].first) && 
						(link->sec2->SequenceNumber == ExcludedLinks[j].second)) || 
						((link->sec2->SequenceNumber == ExcludedLinks[j].first) && 
						(link->sec1->SequenceNumber == ExcludedLinks[j].second)) ){
							cout << "Link " << link->sec1->SequenceNumber<< "-" << link->sec2->SequenceNumber<<"  was excluded" << endl;
							//cin.get();
							badlink = true;
						}
				}
				
				if (badlink == false) {
					LinksList.push_back(link);
				}
			}
		}
	}
	fid.close();
	cout << "Links parsed: " << LinksList.size() << endl;
	return true;
}

void StackRepair::WriteOutputImages( ) {
	
	vector<Section*>::iterator it;
	cout << "Creating output images in " << OutputDirectoryNEW << endl;

	Section *curr = startSection;
	if (curr == NULL) {
		return;
	}

	while ( curr != NULL) {

	
		stringstream ss;
		ss << setw(4)<<setfill('0')<< curr->SequenceNumber<<"_";
	
		fs::path ofname = OutputDirectoryNEW;
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




void StackRepair::WriteTransforms() {
	
	//write all output files, Open ParameterFileOLD file in read mode, ParameterFileNEW and transform file in write mode
	//for every line in ParameterFileOLD file, write the corresponding line in ParameterFileNEW file
	//ParameterFileOLD, ParameterFileNEW
	ifstream fid(ParameterFileOLD.string().c_str(),ios::binary);
	ofstream ofid(ParameterFileNEW.string().c_str(),ios::binary);

	if (fid.good() ==  false) {
		cout << "Parameter file " << ParameterFileOLD.string() << " cannot be opened" << endl;
		return;
	}
	if (ofid.good() ==  false) {
		cout << "Parameter  file " << ParameterFileNEW.string() << " cannot be opened" << endl;
		return;
	}
	
	time_t rawtime;
	struct tm * timeinfo;
	
	time (&rawtime);
	timeinfo = localtime (&rawtime);
	cout << "Writing " << ParameterFileNEW.string() << endl;
	
	ofid << "#Brain:" << brain << endl;
	ofid << "#ComposeDate:" << asctime(timeinfo) ;
	ofid << "#GlobalParameters:"<<GlobalParameters[0]<<","<<GlobalParameters[1]<<","<<GlobalParameters[2]<<endl;
	
	string line;
	
	vector<bool> usedLinks(LinksList.size(),false);
	
	Links *link = new Links();
	Section *sec =  new Section();
	while (fid.good() == true) {
		getline(fid,line);
		//cout << line << endl;
		if (line.length() < 1) {
			continue;
		}
		if (line[0] == '$') {
			parseLinkLine(link,line);
			//sec#1,sec#2,cx,cy,rot,tx,ty,err
			bool found = false;
			for (unsigned int i=0; i<LinksList.size(); ++i) {
				if ((link->sec1 == LinksList[i]->sec1) && (link->sec2 == LinksList[i]->sec2)) {
					TransformType::ParametersType param = LinksList[i]->Tx12->GetParameters();
					ofid << "$" << LinksList[i]->sec1->SequenceNumber << ","
						<< LinksList[i]->sec2->SequenceNumber << "," << param[0] << ","
						<< param[1] << ","<< param[2] << ","
						<< param[3] << ","<< param[4] << ","
						<< LinksList[i]->error << endl;
					//cout << "Link writing from memory" << endl; 
					found  = true;
					usedLinks[i] = true;
					break;
				}
			}
			if (found == false) {
				line.append("\n");
				ofid.write(line.c_str(),line.length());
				cout << "Old link" << endl; 
			}
		}
		else if (line[0] == '#'){
			if (line.find("#GlobalParameters") == string::npos) {
				line.append("\n");
				ofid.write(line.c_str(),line.length());
			}
		}
		else if (line[0] == '0') {
			line.append("\n");
			ofid.write(line.c_str(),line.length());
		}
		else if(line[0] == '1') {
			//scan SectionList and fill up
			//cout <<"parsing line"<<endl;
			parseSectionLine(sec,line);
			int SectionOK = -1;
			//cout << "Sec: " << sec->SequenceNumber << endl;
			Section *matchedSec = NULL;
			for (unsigned int i=0; i<SectionList.size(); ++i) {
				if (SectionList[i]->SequenceNumber == sec->SequenceNumber) {
					SectionOK = (SectionList[i]->Usable == true) ? 1 : 0;
					matchedSec = SectionList[i];
					//matchedSec->Show();
					break;
				}
			}
			if ((SectionOK == 1)&&(matchedSec != NULL) ){
				ofid << "1,"<< matchedSec->SequenceNumber<<","	<< label << 
				"," << matchedSec->SectionName << 
				",Ox=" <<	matchedSec->origin[0] <<
				",Oy=" <<	matchedSec->origin[1] <<
				",Dx=" <<	matchedSec->direction[0] <<
				",Dy=" <<	matchedSec->direction[1] <<
				",Sx=" <<	matchedSec->size[0] <<
				",Sy=" <<	matchedSec->size[1] <<
				",Hf=0,Vf=0,Tf=0," << 
				matchedSec->SectionPath.parent_path().string().c_str() << endl;
			}
			else if ((SectionOK == 0) && (matchedSec != NULL)) {
				ofid << "0,"<< matchedSec->SequenceNumber<<","	<< label << 
				"," << matchedSec->SectionName << 
				",Ox=0,Oy=0,Dx=0,Dy=0,Sx=0,Sy=0,Hf=0,Vf=0,Tf=0," <<
				matchedSec->SectionPath.parent_path().string().c_str() << endl;
			}
			else {
				line.append("\n");
				ofid.write(line.c_str(),line.length());			
			}
			//cout << "STOP" << endl;
			//cin.get();
		}
	}
	delete link;
	delete sec;
	for (unsigned int i=0; i<LinksList.size(); ++i) {
		if (usedLinks[i] == false) {
			TransformType::ParametersType param = LinksList[i]->Tx12->GetParameters();
			ofid << "$" << LinksList[i]->sec1->SequenceNumber << ","
				<< LinksList[i]->sec2->SequenceNumber << "," << param[0] << ","
				<< param[1] << ","<< param[2] << ","
				<< param[3] << ","<< param[4] << ","
				<< LinksList[i]->error << endl;		
		}
	}
	
	fid.close();
	ofid.close();
	
	//Creating the Xform files
	ofstream xfid(TransformFilenameNEW.string().c_str());
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



int main(int argc, char * argv[]) {

	if (argc < 3) {
		cout << "Usage : " << argv[0] << " OLDOutputDirectory NEWOutputDirectory [ options ] " << endl 	<< "Options: " << endl 
		<< "\t -ROT (global rotation in degrees, default: 0)"<< endl
		<< "\t -X (global translation X in pixels, default: 0)"<< endl 
		<< "\t -Y (global translation Y in pixels, default: 0)"<< endl
		<< "\t -gx (canvas size X in pixels, default: 1000)"<< endl
		<< "\t -gy (canvas size Y in pixels, default: 1000)"<< endl
		<< "\t -cmd (repair string in [], e.g. [5-6,7])"<< endl;

		return EXIT_FAILURE;
	}		
	
	double off[3] = {0.0, 0.0, 0.0};
	vector<bool> off_provided(3,false);
	bool UserSuppliedGlobalparams = false;
	double beta_inv = 25.0;
	unsigned int gx = 1000;
	unsigned int gy = 1000;
	int i = 3;
	debugON = false;
	string excl("");
	
	while (i < (argc-1)) {
		string token(argv[i++]);
		string value(argv[i++]);
		if (token.compare("-ROT") == 0) {
			off[0] = vnl_math::pi*atof(value.c_str())/180.0;
			off_provided[0] = true;
			cout << "Global Rotation: " << off[0] << endl;
		}
		else if (token.compare("-beta_inv") == 0) {
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
		else if (token.compare("-X") == 0) {
			off[1] = atoi(value.c_str());
			off_provided[1] = true;
			cout << "Global X: " << off[1] << endl;
		}
		else if (token.compare("-Y") == 0) {
			off[2] = atoi(value.c_str());
			off_provided[2] = true;
			cout << "Global Y: " << off[2] << endl;
		}		
		else if (token.compare("-cmd") == 0) {
			excl = value;
			cout << "cmd: " << excl << endl;
		}		
		else if (token.compare("-debug") == 0) {
			debugON = true;
			cout << endl << "RUNNING IN DEBUG MODE (PRESS ENTER To CONFIRM)" << endl << endl;
			cin.get();
			i--;
		}
		else {
			cout << "Token "<< token <<" cannot be identified" << endl;
			return EXIT_FAILURE;
		}		
	}

	
	StackRepair repr(argv[1], argv[2], gx, gy);
	if (repr.parameter_file_found == false) {
		cout << "Cannot open parameter file in " << argv[1] << " aborting.. " << endl;
		return EXIT_FAILURE;
	}
	
	if (fs::exists(repr.ParameterFileOLD) == false) {
		cout << "Missing Parameter file in the output folder : " << repr.ParameterFileOLD.string() << " aborting.. "<< endl;
		return EXIT_FAILURE;
	}
	
	
	if (excl.length() > 0) {
		if (repr.parseExclusions(excl) == false) {
			cout << "{CMD} cannot be parsed" << endl;
			return EXIT_FAILURE;
		}
	}
	
	if (repr.ReadRecord() == false) {
		return EXIT_FAILURE;
	}
	
	//Find out the source of off parameters
	for (int i=0; i<3;++i) {
		if (off_provided[i] == true) {
			repr.GlobalParameters[i] = off[i];
			if (i == 0) { repr.GlobalParameters[i] *= 180.0/vnl_math::pi;}
		}
		else {
			off[i] = repr.GlobalParameters[i];
			if (i == 0)  {off[i] *= vnl_math::pi/180.0; }
		}
	}

	repr.ResolveLinks();
	repr.ComposeStack(off);
	repr.WriteOutputImages();
	repr.WriteTransforms();
	return EXIT_SUCCESS;
}
