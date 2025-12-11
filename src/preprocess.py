import pandas as pd
import numpy as np
import joblib

class Preprocessor:
    def __init__(self):
        # load all encoders / maps
        self.state_freq_map = joblib.load("../models/encoders/state_freq_map.joblib")
        self.city_freq_map = joblib.load("../models/encoders/city_freq_map.joblib")
        self.locality_target_mean_map = joblib.load("../models/encoders/locality_target_mean_map.joblib")
        self.le_dict = joblib.load("../models/encoders/le_dict.joblib")
        self.mlb_amenities = joblib.load("../models/encoders/mlb_amenities.joblib")
        self.amenity_cols = joblib.load("../models/encoders/amenity_cols.joblib")
    
    def encode_state(self, state): 
        return self.state_freq_map.get(state, 0.0)
    
    def encode_city(self, city): 
        return self.city_freq_map.get(city, 0.0)
    
    def encode_locality(self, state, city, locality): 
        default_val = np.mean(list(self.locality_target_mean_map.values()))
        return self.locality_target_mean_map.get((state, city, locality), default_val)
    
    def encode_labels(self, df):
        """
            encode the leftover columns 👇
            'Property_Type', 'Furnished_Status', 'Public_Transport_Accessibility',
            'Parking_Space', 'Security', 'Facing', 'Owner_Type', 'Availability_Status'
        """

        for col, le in self.le_dict.items():
            df[col] = le.transform(df[col])
        return df
    
    def encode_amenities(self, amenities_list):
        # convert user's list -> ML-ready vector
        amenities_vec = self.mlb_amenities.transform([amenities_list])[0]
        return amenities_vec

    def transform(self, user_input):
        """
        user_input: dict (raw form from Streamlit)
        returns: final feature vector ready for model
        """

        df = pd.DataFrame([user_input])

        # Frequency Encoding 
        df["State_Freq"] = df["State"].apply(self.encode_state)
        df["City_Freq"] = df["City"].apply(self.encode_city)

        # Target Encoding for Locality 
        df["Locality_Target"] = df.apply(
            lambda row: self.encode_locality(row["State"], row["City"], row["Locality"]),
            axis=1
        )

        # Label Encoding
        df = self.encode_labels(df)

        # Amenities Encoding
        amenities_vec = self.encode_amenities(df.loc[0, "Amenities"])
        amenities_df = pd.DataFrame(
            [amenities_vec], 
            columns=self.amenity_cols, 
            index=df.index   # match index with main df
        )

        # drop original categorical columns
        columns_to_drop = [
            "State", "City", "Locality", "Property_Type", "Furnished_Status", 
            "Public_Transport_Accessibility", "Parking_Space", "Security", "Facing", 
            "Owner_Type", "Availability_Status", "Amenities"
        ]
        df_clean = df.drop(columns=columns_to_drop)

        # Final merge
        df_final = pd.concat([df_clean, amenities_df], axis=1)

        return df_final 