# Import the new module
import OAuth_drive_upload_module
import pandas as pd

df = pd.read_csv(
    "status/logmessage.csv",
    header=None,
    names=["Constituency_name", "Log", "upload_status"]
)

print(df)

folder_name_to_process = "চট্টগ্রাম-১১" # Or whatever variable holds the name
my_log_message = "971/964:7 descrepancies"

#3drive_upload_module.process_and_upload_folder(folder_name_to_process, my_log_message)