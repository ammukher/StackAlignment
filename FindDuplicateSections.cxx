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

int main (int argc, char * argv[]) {
	cout << endl<<"Inspecting duplicate and QC on sections" << endl;
	string url("tcp://mitramba1.cshl.edu:3306");
	const string user("registration");
	const string password("admin");
	const string database("MBAStorageDB");
	sql::Driver * driver = sql::mysql::get_driver_instance();
	std::auto_ptr< sql::Connection > con(driver->connect(url, user, password));
	std::auto_ptr< sql::Statement > stmt(con->createStatement());
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
	
	vector<string> BrainList;
	
	if (argc > 1) {
		for (int i=1; i<argc;++i){
			BrainList.push_back(argv[i]);
		}
	}
	else {
		stringstream query1;
		//select name from Navigator_brain where name like "PMD%";
		//SELECT DISTINCT(tracer), BregmaSection, InjectionSection, TargetInjection, ActualInjection FROM Navigator_brain JOIN Navigator_injection ON Navigator_brain.name=Navigator_injection.brain_id WHERE name=\"" + brainName + "\"
		//query1 << "SELECT name FROM Navigator_brain WHERE name LIKE \"PMD%\"";
		query1 << "SELECT name FROM Navigator_brain WHERE name LIKE \"PMD%\" and isQC=1 and isFinalized=1  and onPortal=0";
			
		try {
			std::auto_ptr< sql::ResultSet > res(stmt->executeQuery(query1.str().c_str()));
			cout << "Inspecting " << res->rowsCount() << " brains" << endl;
			while (res->next()) {
				BrainList.push_back(res->getString(1));
			}
		}
		catch (sql::SQLException &e) {
			cout << "SQLException "<< e.what();
			cout << " MySQL error code: " << e.getErrorCode();
			cout << ", SQLState: " << e.getSQLState() << " )" << endl; 
			return false;
		}
		
		/*
		try {
			stringstream query4;
			query4 << "UPDATE Navigator_section SET isDuplicated=0 " << endl;
			stmt->execute(query4.str().c_str());
		}
		catch (sql::SQLException &e) {
			cout << "SQLException "<< e.what();
			cout << " MySQL error code: " << e.getErrorCode();
			cout << ", SQLState: " << e.getSQLState() << " )" << endl; 
			return false;
		}
		*/	
	}
	vector<string> LabelList;
	LabelList.push_back("F");
	LabelList.push_back("IHC");
	LabelList.push_back("N");
	
	int NumberOfDuplicate = 0;
	int MarkedForGoDelete = 0;
	
	//BrainList.clear();
	//BrainList.push_back("PMD788");
	
	for (int k=0; k<BrainList.size(); ++k)	{ 
		
		vector<int> NumSectionsPerLabel(3,0); 
		for (int l=0; l<LabelList.size(); ++l) {
		
			
			stringstream query5;
			query5 << "UPDATE Navigator_section SET isDuplicated=0 WHERE brain_id=\"" << 
								BrainList[k] <<"\" AND label=\"" <<LabelList[l] <<"\"" << endl;
			stmt->execute(query5.str().c_str());
			
		
			stringstream query2;
			query2 << "SELECT modeIndex,isDuplicated,goDelete,OneWeekQC FROM Navigator_section WHERE brain_id=\"" << BrainList[k] << "\" AND label=\"" <<LabelList[l] <<"\" ORDER BY modeIndex" << endl;
		
			try {
				std::auto_ptr< sql::ResultSet > res(stmt->executeQuery(query2.str().c_str()));
				
				NumSectionsPerLabel[l] = res->rowsCount();
				
				if (res->rowsCount() == 0) {
					continue;
				}
				vector<int> goDeleteList,isDuplicatedList;
				int goDelete, OneWeekQC = 0;
				int isDuplicated;
				int ModeNumber, lastModeNumber = -1, lastlastModeNumber = -1;
				int cnt = 1, goDelCnt = 0;
				int SeriesWiseDuplicate = 0;
				int SeriesWiseDataInQC = 0;
				while (res->next()) {
					ModeNumber = res->getInt(1);
					isDuplicated = res->getInt(2);
					goDelete = res->getInt(3);
					OneWeekQC = res->getInt(4);
					
					if (OneWeekQC > 0) {
						SeriesWiseDataInQC++;
					}
					
					if (lastModeNumber != ModeNumber) {
						if (cnt > 1) {
							if ((cnt - goDelCnt)>1) {
								//marking lastModeNumber for Duplicate
								stringstream query3;
								query3 << "UPDATE Navigator_section SET isDuplicated=1 WHERE brain_id=\"" << 
								BrainList[k] <<"\" AND label=\"" <<LabelList[l] <<"\" AND modeIndex=" <<
								lastModeNumber << endl;
								stmt->execute(query3.str().c_str());
								//cout << query3.str();
								NumberOfDuplicate++;
								SeriesWiseDuplicate++;
								//cin.get();
							}
							else if ((cnt - goDelCnt) == 1) {
								// clear isDuplicate flag from lastModeNumber
								stringstream query3;
								query3 << "UPDATE Navigator_section SET isDuplicated=0 WHERE brain_id=\"" << 
								BrainList[k] <<"\" AND label=\"" <<LabelList[l] <<"\" AND modeIndex=" <<
								lastModeNumber << endl;	
								stmt->execute(query3.str().c_str());
								MarkedForGoDelete++;								
								//cout << query3.str();								
								//cin.get();
							}
						}
						cnt = 1;
						goDelCnt = (goDelete==1) ? 1 : 0;
					}
					else {
						cnt++;
						goDelCnt += (goDelete==1) ? 1 : 0;
					}
					lastlastModeNumber = lastModeNumber;
					lastModeNumber = ModeNumber;
				}
				
				// Do once more for the last section
				if (lastModeNumber != lastlastModeNumber) {
					if (cnt > 1) {
						if ((cnt - goDelCnt)>1) {
							//marking lastModeNumber for Duplicate
							stringstream query3;
							query3 << "UPDATE Navigator_section SET isDuplicated=1 WHERE brain_id=\"" << 
							BrainList[k] <<"\" AND label=\"" <<LabelList[l] <<"\" AND modeIndex=" <<
							lastModeNumber << endl;
							stmt->execute(query3.str().c_str());
							//cout << query3.str();
							NumberOfDuplicate++;
							SeriesWiseDuplicate++;
							//cin.get();
						}
						else if ((cnt - goDelCnt) == 1) {
							// clear isDuplicate flag from lastModeNumber
							stringstream query3;
							query3 << "UPDATE Navigator_section SET isDuplicated=0 WHERE brain_id=\"" << 
							BrainList[k] <<"\" AND label=\"" <<LabelList[l] <<"\" AND modeIndex=" <<
							lastModeNumber << endl;	
							stmt->execute(query3.str().c_str());
							MarkedForGoDelete++;								
							//cout << query3.str();								
							//cin.get();
						}
					}
					cnt = 1;
					goDelCnt = (goDelete==1) ? 1 : 0;
				}
				else {
					cnt++;
					goDelCnt += (goDelete==1) ? 1 : 0;
				}				
				
				
				if ((SeriesWiseDuplicate > 0) || (SeriesWiseDataInQC > 0)) {
					cout << "Number of duplicate sections for " << BrainList[k]<<" "<< LabelList[l] <<": " ;
					cout << SeriesWiseDuplicate << ", in QCdisk : " << SeriesWiseDataInQC << endl;
				}
				else {
					cout <<  BrainList[k]<<" "<< LabelList[l] << " good to go with " << NumSectionsPerLabel[l] << " sections." <<  endl;
				}
			}
			catch (sql::SQLException &e) {
				cout << "SQLException "<< e.what();
				cout << " MySQL error code: " << e.getErrorCode();
				cout << ", SQLState: " << e.getSQLState() << " )" << endl;
				return false;
			}
		}
		if ((NumSectionsPerLabel[0] == 0) && (NumSectionsPerLabel[1] == 0) && (NumSectionsPerLabel[2] == 0)) {
            cout << "Brain " << BrainList[k] << " not in Storage !!! " << endl;
        } 
		
	}
	cout << "Total number of Duplicates: " << NumberOfDuplicate << endl;
	cout << "Total number of Duplicate Sections marked for GoDelete: " << MarkedForGoDelete << endl;
	stmt.reset(NULL);
	return 0; 
}
