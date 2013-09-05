#include <iostream>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <unistd.h>
#include <sys/stat.h>

#include <iterator>
#include <algorithm>
#include <sys/time.h>
#include <boost/filesystem.hpp>

#define MAXJOB 200
#define LOGGING 0

using namespace std;
namespace fs = boost::filesystem;
 
string logfilename;
stringstream ss;

#define S(x) cout<<x<<endl;cin.get(); 
int SubmitJobs(fs::path& RegisterFile, unsigned int batch_size, unsigned int);
void getScriptFileName(string& scriptFile, string& line);
//void ReassignPathForBnB(string& line);
const string LOGROOT("/data/mitra/amukherj/XFormLOG/");
const string  outFolder("/data/mitra2/PORTALJP2");
//const string LOGROOT("/home/amit/StackAlign/build/");

#define LOG(x) ss << x; logdata(ss);
void logdata(stringstream& ss) {
	cout << ss.str() << endl;
	time_t rawtime;
	time ( &rawtime );
	string timestamp(ctime(&rawtime));
	timestamp.erase(timestamp.end()-1);
	ofstream lid(logfilename.c_str(),ios::app);
	lid << timestamp << " - " << ss.str()<<endl;
	lid.close();
	ss.str(string(""));
}

int  main(int argc, char **argv) { 
	if ((argc < 2) || (argc > 4)){
		cout << " Usage: " << argv[0] << " Xform_File [batchsize]" <<  endl;
		return EXIT_FAILURE;
	} 
	fs::path RegisterFile(argv[1]);
	unsigned int batch_size = 30;
	if (argc == 3) {
		batch_size = atoi(argv[2]);
	}
	unsigned int group_size = 10;
	time_t rawtime;
	struct tm * timeinfo;
	char logfname [80];
	time ( &rawtime );
	timeinfo = localtime ( &rawtime );
	strftime (logfname,80,"Transform_LOG_%b%d.txt",timeinfo);
	//logfilename = RegisterFile.string().substr(0,RegisterFile.string().length()-4);
	logfilename = LOGROOT;
	logfilename.append(logfname);
	//logfilename.append(logfname);
	
	ss.str(string(""));
	LOG("Starting Executor batch management " << endl << "SectionListFile " <<  RegisterFile << endl << "BatchSize: " << batch_size)
	cout << "LOG FILE: " << logfilename.c_str() << endl;
	
	if (exists(RegisterFile) == false ) {
		cout << RegisterFile << " does not exist " << endl;
		return EXIT_FAILURE;
	}
	
	struct timeval begintime, endtime;
	gettimeofday(&begintime, NULL);
	int success = SubmitJobs(RegisterFile, batch_size,group_size );
	gettimeofday(&endtime, NULL);
	double elapsedtime = (endtime.tv_sec - begintime.tv_sec + 0.001);
	LOG("Batch Complete. Elapsed Time: " << elapsedtime << endl )
	return success;
}	

int SubmitJobs(fs::path& RegisterFile, unsigned int batch_size, unsigned int group_size) {
	unsigned int current_index;
	fs::path RegisterIndexfile = RegisterFile;
	RegisterIndexfile += "_Index";
	//read the current index
	ifstream cid(RegisterIndexfile.string().c_str(),ios::binary);
	char buf[4];
	if (cid.is_open() == true) {
		cid.read(buf,sizeof(int));
		current_index = *((int*)buf);
		cid.close();
	}
	else {
		cout << "Restarting the queue " << endl;
		//clear outfolder and set count directory
		string BrainName = RegisterFile.string();
		size_t q1 = BrainName.find_first_of("_");
		if (q1 == string::npos) {
			LOG("Register filename format is not correct (no underscore) " << RegisterFile)
			return 1;
		}
		BrainName.erase(q1);
		fs::path BrainPath = outFolder;
		BrainPath /= BrainName;
		if (fs::exists(BrainPath) == false) {
			//fs::remove_all(BrainPath);
			fs::create_directory(BrainPath);
		}

		current_index = 1;	
		fs::path countFile = BrainPath;
		countFile /= RegisterFile.string().substr(0,RegisterFile.string().length()-4);
		countFile += "_targetcount.txt";
	
		std::ifstream qid; qid.open(RegisterFile.string().c_str());
		if(qid.is_open() == false) {
			LOG("Register file cannot be open " << RegisterFile)
			return 1; //fail
		}
		std::stringstream qbuffer;
		qbuffer << qid.rdbuf();
		string qb = qbuffer.str();
		std::size_t qn = qb.find_first_of('\n');
		int qcount = 0;
		while(qn != string::npos) {
			qn = qb.find_first_of('\n',qn+1);
			qcount++;
		}
		ofstream qcid(countFile.string().c_str(),ios::binary);
		qcid << qcount<<endl;
		qcid.close();
		qid.close();
	}

	LOG("Current starting index " << current_index )
	//read the actual file	
	std::ifstream fid; fid.open(RegisterFile.string().c_str());
	if(fid.is_open() == false) {
		LOG("Register file cannot be open " << RegisterFile)
		return 1; //fail
	}	

	std::string JobSubmitted = LOGROOT + RegisterFile.filename().string();
	JobSubmitted.erase(JobSubmitted.length()-4 );
	JobSubmitted.append("_JobsONBnB.txt");

	string execCode("/data/mitra/amukherj/XFormCode/ApplyTForm");
	if (RegisterFile.string().find("_F_") != string::npos) {
		execCode.append("16bit");
	}
	else if (RegisterFile.string().find("_N_") != string::npos) {
		execCode.append("8bit");
	}
	else if (RegisterFile.string().find("_IHC_") != string::npos) {
		execCode.append("8bit");
	}
	else {
		LOG("Cannot parse the RegFilename to derive the LabelType " << RegisterFile)
		return 1; //fail
	}
	LOG("Execution Code" << execCode)

	std::stringstream buffer;
	buffer << fid.rdbuf();
	std::string b = buffer.str();
	std::size_t n = b.find_first_of('\n');
	if (n == string::npos) {
		LOG("No token found" )
		return 1;
	}
	
	std::size_t n1 = 0; n = 0;
	for (unsigned int i=1; i<current_index; ++i) {
		n = b.find_first_of('\n',n+1);
		if (n == string::npos) {
			LOG("Current index is past the last token" )
			return 1;
		}
		n1 = n+1;
	}
	
	for (unsigned int i = 0; i < batch_size; ++i) {

		std::stringstream ss1;
		ss1 << "$TMPDIR/JOB" << std::setw(6) << std::setfill('0') << rand()%100000 ;
		std::string tmpdir = ss1.str();

		stringstream data;
		data << "#!/bin/bash" << std::endl <<
		"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/mitra/amukherj/Kakadu/v71:/data/mitra/amukherj/Kakadu/" << std::endl <<
		"export PATH=$PATH:/data/mitra/amukherj/Kakadu/v71:/data/mitra/amukherj" << std::endl <<
		"mkdir " << tmpdir << std::endl << 
		"cd " << tmpdir << std::endl;
		
		unsigned int group_count = 0;
		string lastline;
		for (unsigned int j = 0; j < group_size; ++j )	{
			n = b.find_first_of('\n',n1);
			if (n == string::npos) {
				LOG("Current index is past the last token" )
				break;
			}
			current_index++;
			std::string line = b.substr(n1,n-n1);
			n1 = n+1;
			cout << line << endl;
			//cin.get();
				
			data << execCode << " \""<< line <<"\" " << endl ;
			lastline = line;
			group_count++;
		}
		if (group_count == 0) {
			break;
		}
		
		cout << data.str() << endl << "Number in the group (" << group_count <<")" << endl;
		
		string scriptFile;
		getScriptFileName(scriptFile,lastline);
		cout << "Script file name: " << scriptFile << endl;
		std::ofstream fid(scriptFile.c_str());
		if (!fid.is_open()) {
			std::cerr << "File cannot be opened:" << scriptFile <<" (write permissions)" << std::endl;
			continue;
		}
		fid.write(data.str().c_str(),data.str().length());
		fid.close(); 
		
		
		std::string cmd; cmd.clear();
		#if LOGGING
		cmd.assign("qsub -pe threads 4 -l tmp_free=3.7G -l m_mem_free=3.5G -e /data/mitra/amukherj/error -o /data/mitra/amukherj/output ");
		#else
		cmd.assign("qsub -pe threads 4 -l tmp_free=3.7G -l m_mem_free=3.5G -e /dev/null -o /dev/null ");
		#endif

		cmd.append(scriptFile);
		cout << cmd << endl;
		
		// SUBMITTING JOB here
		if (system(cmd.c_str()) != 0) { 
			cout << "Job submission failed!!" << endl;	
			return 1;	
		}
		remove(scriptFile.c_str());
		
		
		//open job submitted file and make entry
		LOG("Submitted "<< scriptFile)
		std::ofstream lid(JobSubmitted.c_str(),std::ios::app);
		lid << "Job group of size" << group_count << endl << lastline << endl; 
		lid.close();
	}
	
	ofstream did(RegisterIndexfile.string().c_str(),ios::binary);
	if (did.is_open() == true) {
		did.write((char*)&current_index,sizeof(int));
		did.close();
	}
	return 0;
}

void getScriptFileName(string& scriptFile, string& line) {
	scriptFile.assign("JOBGR.sh");
	size_t n1 = line.find("main/M"), n2 = string::npos, n3 = string::npos;
	if (n1 != string::npos) {
		n2 = line.find("/",n1+5);
		n3 = line.find_first_of(",");
	}
	if ((n2 != string::npos) && (n3 != string::npos)) {
		string section = line.substr(0,n3);
		string mdrive = line.substr(n1+5,n2-n1-5);
		//PMD1067&1066-F1-2012.08.05-09.01.47_PMD1067_2_0002
		n1 = section.find_first_of("-");
		n2 = section.find_first_of("_");
		n3 = section.find_first_of("_",n2+1);
		scriptFile = 	string("JOBGR_") + mdrive + string("_") + 
						section.substr(n2+1,n3-n2-1) + string("_") + 
						section.substr(n1+1,1) + string("_") + 
						section.substr(section.length()-4,4) + string(".sh");
	}
}

