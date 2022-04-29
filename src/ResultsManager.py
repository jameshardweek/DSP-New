import pandas as pd
from os.path import exists
from random import choice

class ResultsManager:
    col_names = ["MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","MDVP:Jitter(%)","MDVP:Jitter(Abs)","MDVP:RAP","MDVP:PPQ","Jitter:DDP","MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5","MDVP:APQ","Shimmer:DDA","NHR","HNR","status","RPDE","DFA","spread1","spread2","D2","PPE"]

    def __init__(self, results_file) -> None:
        if '.csv' not in results_file:
            results_file += '.csv'
            
        if exists(results_file):
            try:
                self.results = pd.read_csv(results_file, index_col=0).to_dict(orient='index')
                self.results = {str(k): v for k, v in self.results.items()}
                self.path = results_file
            except:
                print(f"ERROR: Could not parse file {results_file}. Creating empty results table.")
                self.results = {}
                self.path = results_file
        else:
            print(f"ERROR: File {results_file} does not exist. Using defaults.")
            self.results = {}
            self.path = results_file

    # Used to add results to the results table, status can be optionally given
    def add_results(self, features: dict, status=None) -> str:
        uid = str(choice([x for x in range(1,1000) if x not in self.results]))
        self.results[uid] = {}
        for key in self.col_names:
            if key in features:
                self.results[uid][key] = features[key]
            else:
                self.results[uid][key] = None
        self.results[uid]['status'] = status
        return uid

    # Get status for given UID
    def get_status(self, uid: str) -> int:
        if uid in self.results:
            try:
                return int(self.results[uid]['status'])
            except:
                return -1
        else:
            return None

    # Get features for given UID
    def get_features(self, uid: str) -> dict:
        if uid in self.results:
            return {k: v for k, v in self.results[uid].items() if k != 'status'}

    # Update the status for a given UID
    def set_status(self, uid: str, status: bool) -> None:
        if uid in self.results:
            self.results[uid]['status'] = int(status)

    # Delete results for given UID
    def remove_results(self, uid: str) -> None:
        if uid in self.results:
            del(self.results[uid])

    # Returns a list of UIDs without predictions
    def get_unpredicted(self) -> list:
        unpredicted = []
        for uid in self.results:
            if pd.isnull(self.results[uid]['status']):
                unpredicted.append(uid)
        return unpredicted

    # Converts current results dictionary to a pandas dataframe
    def to_dataframe(self, results=None) -> pd.DataFrame:
        if results is None:
            return pd.DataFrame.from_dict(self.results, orient='index').sort_index()
        else:
            return pd.DataFrame([results.values()], columns=results.keys())

    # Outputs a .csv file of results. Defaults to the current working file, else a given output file can be written to
    def save(self, file_path=None) -> None:
        if file_path is None:
            self.to_dataframe().to_csv(self.path, index_label='name')
        else:
            self.to_dataframe().to_csv(file_path, index_label='name')


    # UNUSED FUNCTIONS

    # Used to return all features from a list of UIDs
    # def get_features(self, uids=None):
    #     ret_features = {}
    #     if uids is None:
    #         for uid in self.results:
    #             ret_features[uid] = {x: self.results[uids][x] for x in self.results[uids] if x != 'status'}
    #     else:
    #         if type(uids) is str:
    #             uid = uids
    #             if uid in self.results:
    #                 return {x: self.results[uid][x] for x in self.results[uid] if x != 'status'}
    #         elif type(uids) is list:
    #             for uid in uids:
    #                 if uid in self.results:
    #                     ret_features[uid] = {x: self.results[uid][x] for x in self.results[uid] if x != 'status'}
    #                 else:
    #                     print(f"UID {uid} not in results.")

    #     print()
    #     return ret_features
    
    # Used to get the statuses from a UID or list of UIDs
    # def get_status(self, uids=None):
    #     ret_results = {}
    #     if uids is None:
    #         for uid in self.results:
    #             ret_results[uid] = self.results[uid]['status']
    #     else:
    #         if type(uids) is str:
    #             uid = uids
    #             if uid in self.results:
    #                 return self.results[uid]['status']
    #         elif type(uids) is list:
    #             for uid in uids:
    #                 if uid in self.results:
    #                     ret_results[uid] = self.results[uid]['status']
    #                 else:
    #                     print(f"UID {uid} not in results.")

    #     print()
    #     return ret_results

    # Removes results for a given UID
    # def remove_results(self, uids):
    #     if type(uids) is str:
    #         uid = uids
    #         if uid in self.results:
    #             del(self.results[uids])
    #         else:
    #             print(f"UID {uid} not in results.")
    #     elif type(uids) is list:
    #         for uid in uids:
    #             if uid in self.results:
    #                 del(self.results[uid])
    #             else:
    #                 print(f"UID {uid} not in results.")