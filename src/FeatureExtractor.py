import pandas as pd
import parselmouth as pm


class FeatureExtractor:
    praat_dataset_labels = dict(zip(
        ['Mean pitch','Maximum pitch','Minimum pitch','Jitter (local)','Jitter (local, absolute)','Jitter (rap)','Jitter (ppq5)','Jitter (ddp)','Shimmer (local)','Shimmer (local, dB)','Shimmer (apq3)','Shimmer (apq5)','Shimmer (apq11)','Shimmer (dda)','Mean noise-to-harmonics ratio','Mean harmonics-to-noise ratio'],
        ["MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","MDVP:Jitter(%)","MDVP:Jitter(Abs)","MDVP:RAP","MDVP:PPQ","Jitter:DDP","MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5","MDVP:APQ","Shimmer:DDA","NHR","HNR"], 
    ))
    dataset_labels = ["MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","MDVP:Jitter(%)","MDVP:Jitter(Abs)","MDVP:RAP","MDVP:PPQ","Jitter:DDP","MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5","MDVP:APQ","Shimmer:DDA","NHR","HNR","RPDE","DFA","spread1","spread2","D2","PPE"]

    def __init__(self, sound_data) -> None:
        self.features = {}

        sound = pm.Sound(sound_data)
        pitch = sound.to_pitch()
        pulses = pm.praat.call([sound, pitch], "To PointProcess (cc)")
        voice_report = pm.praat.call([sound, pitch, pulses], "Voice report", 2,0,75,500,1.3,1.6,0.03,0.45)

        voice_report_dict = dict(item.strip().split(": ") for item in voice_report.split("\n")[1:] if ": " in item)

        for key, value in voice_report_dict.items():
            if value == '--undefined--':
                continue
            new_value = ''.join(s for s in value if s.isdigit() or s == '.' or s == '%' or s == 'E' or s == '-' or s == '(')
            if '(' in new_value:
                new_value = new_value.split('(')[0]
            if '%' in new_value:
                voice_report_dict[key] = float(new_value.replace('%','')) / 100
            else:
                voice_report_dict[key] = float(new_value)

        relevant_features = {}
        for key, value in self.praat_dataset_labels.items():
            voice_report_dict[value] = voice_report_dict[key]

        voice_report_dict['RPDE'] = self.calculate_rpde()
        voice_report_dict['DFA'] = self.calculate_dfa()
        voice_report_dict['spread1'] = self.calculate_spread1()
        voice_report_dict['spread2'] = self.calculate_spread2()
        voice_report_dict['D2'] = self.calculate_d2()
        voice_report_dict['PPE'] = self.calculate_ppe()

        self.features = voice_report_dict

    def calculate_rpde(self) -> float:
        return None

    def calculate_dfa(self) -> float:
        return None

    def calculate_spread1(self) -> float:
        return None

    def calculate_spread2(self) -> float:
        return None

    def calculate_d2(self) -> float:
        return None

    def calculate_ppe(self) -> float:
        return None

    def get_features(self, features=None) -> dict:
        if features is None:
            return self.features
        else:
            return {k: v for k, v in self.features.items() if k in features}

    # Returns dict of features present in the dataset
    def get_dataset_features(self) -> dict:
        return {k:v for k,v in self.features.items() if k in self.dataset_labels}

    # Converts features to dataframe
    def to_dataframe(self, features=None) -> pd.DataFrame:
        if features is None:
            return pd.DataFrame([self.features.values()], columns=self.features.keys())
        else:
            return pd.DataFrame([features.values()], columns=features.keys())

    # Get features for specific sounds or features
    # If fetch_sounds is passed, return all features for those sounds
    # If fetch_features is passed, return those features for all sounds
    # If both fetch_sounds and fetch_features are passed, return only specified features for specified sounds
    # def get_features(self, fetch_sounds=None, fetch_features=None):
    #     ret_features = {}

    #     if fetch_sounds is None and fetch_features is None:
    #         return self.features

    #     if fetch_features is None:
    #         for sound in fetch_sounds:
    #             if sound in self.features:
    #                 ret_features[sound] = self.features[sound]
    #             else:
    #                 print(f"Sound {sound} not present in database.")

    #         return ret_features

    #     if fetch_sounds is None:
    #         for sound in self.features:
    #             ret_features[sound] = {}
    #             for feature in fetch_features:
    #                 if feature in self.features[sound]:
    #                     ret_features[sound][feature] = self.features[sound][feature]
    #                 else:
    #                     print(f"Feature {feature} not present for sound {sound}")

    #         return ret_features

    #     for sound in fetch_sounds:
    #         if sound in self.features:
    #             ret_features[sound] = {}
    #             for feature in fetch_features:
    #                 if feature in self.features[sound]:
    #                     ret_features[sound][feature] = self.features[sound][feature]
    #                 else:
    #                     print(f"Feature {feature} not present for sound {sound}")
    #         else:
    #             print(f"Sound {sound} not present in database.")

    #     print()
    #     return ret_features

    # Returns dict of features for all features present in the dataset
    # def get_dataset_features(self, sounds=None) -> dict:
    #     ret_features = {}
    #     if sounds is None:
    #         for sound in self.features:
    #             ret_features[sound] = {x:None for x in self.dataset_labels}
    #             for feature in self.dataset_labels:
    #                 if feature in self.features[sound]:
    #                     ret_features[sound][feature] = self.features[sound][feature]
    #                 else:
    #                     print(f"Feature {feature} not present for sound {sound}")

    #     else:
    #         if type(sounds) is list:
    #             for sound in sounds:
    #                 if sound in self.features:
    #                     ret_features[sound] = {x:None for x in self.dataset_labels}
    #                     for feature in self.dataset_labels:
    #                         if feature in self.features[sound]:
    #                             ret_features[sound][feature] = self.features[sound][feature]
    #                         else:
    #                             print(f"Feature {feature} not present for sound {sound}")
    #                 else:
    #                     print(f"Sound {sound} not present in database.")
    #         elif type(sounds) is str:
    #             sound = sounds
    #             if sound in self.features:
    #                 ret_features = {x:None for x in self.dataset_labels}
    #                 for feature in self.dataset_labels:
    #                     if feature in self.features[sound]:
    #                         ret_features[feature] = self.features[sound][feature]
    #                     else:
    #                         print(f"Feature {feature} not present for sound {sound}")
    #             else:
    #                 print(f"Sound {sound} not present in database.")

    #     print()
    #     return ret_features

    # Gets all features for all extracted sounds. If argument is given, return features for only that sound
    # def get_all_features(self, sound=None):
    #     ret_features = {}
    #     if sound is None:
    #         ret_features = self.features
    #     else:
    #         if sound in self.features:
    #             ret_features[sound] = self.features[sound]
    #         else:
    #             print(f"Sound {sound} not in features dict.")

    #     print()
    #     return ret_features
