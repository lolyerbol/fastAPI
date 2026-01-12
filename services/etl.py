import pandas as pd

def remove_outliers(new_frame):
    new_frame['trip_duration_minutes'] = round((new_frame['tpep_dropoff_datetime'] - new_frame['tpep_pickup_datetime']).dt.total_seconds()/60,2)

    a = new_frame.shape[0]
#    print("Number of pickup records:", a)

    temp_frame = new_frame[(new_frame.trip_duration_minutes > 0)&(new_frame.trip_duration_minutes <720)]
    c = temp_frame.shape[0]
#    print("Number of outliers from trip times analysis:", (a-c))
    
    temp_frame = new_frame[(new_frame.trip_distance > 0) & (new_frame.trip_distance < 40)]
    d = temp_frame.shape[0]
#    print ("Number of outliers from trip distance analysis:", (a-d))
    
    temp_frame = new_frame[(new_frame.total_amount < 500) & (new_frame.total_amount > 0)]
    e = temp_frame.shape[0]
#    print ("Number of outliers from fare analysis:", (a-e))

    new_frame = new_frame[(new_frame.trip_duration_minutes > 0)&(new_frame.trip_duration_minutes <720)]
    new_frame = new_frame[(new_frame.trip_distance > 0) & (new_frame.trip_distance < 40)]
    new_frame = new_frame[(new_frame.total_amount < 500) & (new_frame.total_amount > 0)]


    return new_frame


def clean_df(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input DataFrame by handling missing values and converting data types.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame to be cleaned."""
    # 1. Fill missing values
    data = data.fillna({
        'passenger_count': 1,
        'trip_distance': 0.0,
        'fare_amount': 0.0,
        'total_amount': 0.0,
        'RatecodeID': 99
    })
    # 2. Convert data types
    data['passenger_count'] = data['passenger_count'].astype(int)
    data['trip_distance'] = data['trip_distance'].astype(float)
    data['fare_amount'] = data['fare_amount'].astype(float)
    data['total_amount'] = data['total_amount'].astype(float)
    data['vendor_id'] = data['VendorID'].astype(int)
    data['ratecode_id'] = data['RatecodeID'].astype(int)
    data["payment_type"] = data["payment_type"].astype(int)
    data.rename(columns={'PULocationID': 'pickup_location_id', 'DOLocationID': 'dropoff_location_id'},inplace=True)

    # 3. Drop unnecessary columns
    data = data.drop(columns=['congestion_surcharge', 'Airport_fee', 'cbd_congestion_fee','VendorID','RatecodeID'], errors='ignore')
#    data = remove_outliers(data)
    data = remove_outliers(data)
    data = data.dropna(subset=['store_and_fwd_flag'])
    print("Without nulls",data.shape[0])
    return data



"""
payment_type = pd.DataFrame({"pyament_type": [0,1,2,3,4,5,6],
                             "payment_type_name":[ "Flex Fare trip","Credit card","Cash","No charge","Dispute","Unknown","Voided trip"]})


vendor = pd.DataFrame({"vendor_id": [1,2,6,7],
                    "vendor_name": [ " Creative Mobile Technologies, LLC", "Curb Mobility, LLC","Myle Technologies Inc","Helix"]})

rate_code_id = pd.DataFrame({"rate_code_id": [1,2,3,4,5,6,99], "rate_code_name": ["Standard rate","JFK","Newark","Nassau or Westchester","Negotiated fare","Group ride","Null/unknown "]})"""