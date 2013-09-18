#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <ctime>
#include <cmath>
#include <vector>
#include <iomanip>

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
/*#include <cppconn/connection.h"*/
#include "mysql_driver.h"
#include "mysql_connection.h"

using namespace std;
namespace fs = boost::filesystem;

int main(int argc, char* argv[]) {
	if (argc != 2) {
		cout << "Insufficient arguments " << endl;
		cout << "Usage :"  << argv[0] << " BrainName" << endl;
		return EXIT_FAILURE;
	}
	string BrainName(argv[1]);
	
	cout << endl<<"Inspecting " << BrainName << endl;
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
	cout << "Database connection success" << endl <<endl;	
	
	const int NumLabels = 3;
	vector<string> labelList;
	labelList.push_back(string("F"));
	labelList.push_back(string("IHC"));
	labelList.push_back(string("N"));
	

	for (int label=0; label<NumLabels; ++label) {
		
		stringstream queryF;
		vector<string> SectionList;
		
		queryF << "SELECT filename,path,modeIndex,misLabeled,isDamaged,reCoverslip,reImage,goDelete FROM Navigator_section WHERE brain_id=\"" 
		<< BrainName << "\" AND label=\""<<labelList[label]<<"\" ORDER BY modeIndex;" << endl;
		
		//cout << queryF.str().c_str() << endl;
		try {
			std::auto_ptr< sql::ResultSet > res(stmt->executeQuery(queryF.str().c_str()));
			cout << "Number of sections for " << labelList[label] <<": " << res->rowsCount() << endl;
			int cnt = 0, valid = 0;
			//int bestSecId = 0;
			int lastModeNumber = -1;
			while (res->next()) {
				cnt++;
				string SectionName = res->getString(1);
				string filepath = res->getString(2);
				int ModeNumber = res->getInt(3);
				int Usable = 0;
				for (int i=4; i<=8;++i) {
					Usable +=  res->getInt(i);
				}
				
				if (Usable > 0) {
					continue;
				}
				if (lastModeNumber == ModeNumber) 	{
					cout << "Warning: Duplicate sections ModeIndex: " << lastModeNumber << " and " << ModeNumber << endl; 
				}
				
				filepath.erase(filepath.length()-1);
								
				//test logic: lossless JP2 not in original folder
				fs::path losslessJP2 = filepath;
				losslessJP2 /= SectionName;
				losslessJP2 += "_lossless.jp2";
				if (fs::exists(losslessJP2) == false) {
					filepath.append("_JP2");	
				}			
				filepath.append("/");
				filepath.append(SectionName);
				filepath.append(".tif");
				valid++;
				
				SectionList.push_back(filepath);			
				lastModeNumber = ModeNumber;
			}	
		}

		catch (sql::SQLException &e) {
			cout << "SQLException "<< e.what();
			cout << " MySQL error code: " << e.getErrorCode();
			cout << ", SQLState: " << e.getSQLState() << " )" << endl;
			return false;
		}
		queryF.str("");
		
		cout << "Number of usable sections " << SectionList.size() << endl;
		if (SectionList.size() > 0) {
			string outfname(BrainName);
			outfname.append("_");
			outfname.append(labelList[label]);
			outfname.append("_List.txt");
			cout << "Writing " << outfname << endl;
			ofstream fid(outfname.c_str(),ios::binary);
			for (unsigned int i=0; i<SectionList.size(); ++i) {
				fid << SectionList[i] << endl;
			}
			fid.close();
		}	
	}
	
	return EXIT_SUCCESS;
}