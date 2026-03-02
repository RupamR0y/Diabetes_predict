import pandas as pd
import numpy as np
import os
import time
import sys
import re
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

DATABASE_FILE = "patient_logs.csv"
SESSION_DATE = None 

class AbortEntry(Exception):
    pass

def clear():
    os.system('clear')

# --- 1. PREPARE DATA & LOADING SCREEN ---
clear()
print("[*] Loading medical data and training model... Please wait.")

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Preg', 'Gluc', 'BP', 'Skin', 'Insu', 'BMI', 'DPF', 'Age', 'Outcome']

try:
    data = pd.read_csv(url, names=columns)
except Exception as e:
    print(f"\n(!) Error downloading data: {e}")
    exit()

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = GaussianNB()
model.fit(X_scaled, y)

print("System Initializing", end="")
sys.stdout.flush()
for _ in range(5):
    time.sleep(1)
    print(".", end="")
    sys.stdout.flush()
clear()

# --- 2. INPUT VALIDATION FUNCTIONS ---
def get_string(prompt, enforce_space=False, enforce_digit=False):
    while True:
        val = input(prompt).strip()
        if val.lower() == 'exit':
            raise AbortEntry()
        if not val:
            print("\n(!) Invalid input type again ...\n")
            continue
        if enforce_space and ' ' not in val:
            print("\n(!) Please enter full name with a space between different words.\n")
            continue
        if enforce_digit and not val.isdigit():
            print("\n(!) Invalid input type again ...\n")
            continue
        return val

def get_float(prompt):
    while True:
        val = input(prompt).strip()
        if val.lower() == 'exit':
            raise AbortEntry()
        if not val:
            print("\n(!) Invalid input type again ...\n")
            continue
        try:
            return float(val)
        except ValueError:
            print("\n(!) Invalid input type again ...\n")

def get_age(prompt):
    while True:
        val = input(prompt).strip()
        if val.lower() == 'exit':
            raise AbortEntry()
        if not val:
            print("\n(!) Invalid input type again ...\n")
            continue
        try:
            num = int(val)
            if 1 <= num <= 150:
                return num
            else:
                print("\n(!) Wrong age ..type again ...\n")
        except ValueError:
            print("\n(!) Invalid input type again ...\n")

def get_date(prompt):
    while True:
        val = input(prompt).strip()
        if val.lower() == 'exit':
            raise AbortEntry()
        if not val:
            print("\n(!) Invalid input type again ...\n")
            continue
        
        if re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", val):
            d, m, y = map(int, val.split('/'))
            
            if not (2025 <= y <= 2125) or not (1 <= m <= 12):
                print("\n(!) Please enter the current date , type again ...\n")
                continue
            
            if m == 2:
                if y % 400 == 0:
                    max_d = 29
                elif y % 100 == 0:
                    max_d = 28
                elif y % 4 == 0:
                    max_d = 29
                else:
                    max_d = 28
            elif m in [1, 3, 5, 7, 8, 10, 12]: 
                max_d = 31
            else: 
                max_d = 30
                
            if not (1 <= d <= max_d):
                print("\n(!) Please enter the current date , type again ...\n")
                continue
            
            return f"{d:02d}/{m:02d}/{y}"
            
        print("\n(!) Invalid input type again ...\n")

# --- 3. DATABASE HELPER FUNCTIONS ---
def check_duplicate_id(p_id, full_name):
    if not os.path.exists(DATABASE_FILE) or os.stat(DATABASE_FILE).st_size == 0:
        return True 
        
    df = pd.read_csv(DATABASE_FILE)
    df['ID'] = df['ID'].astype(str)
    
    if p_id in df['ID'].values:
        matches = df[df['ID'] == p_id]
        row = matches.iloc[-1] 
        existing_name = f"{row['First Name']} {row['Last Name']}"
        
        if existing_name.strip().lower() == full_name.strip().lower():
            print(f"\n(!) Warning: Data for {existing_name.title()} already existed.\n")
            print("--- Most Recent Matched Person Details ---")
            print(f"ID     : {row['ID']}")
            print(f"Name   : {existing_name.title()}")
            print(f"Gender : {row['Gender']}")
            print(f"Age    : {int(row['Age'])}")
            print(f"Date   : {row['Date']}")
            print("------------------------------------------\n")
            
            while True:
                ans = input("Is this the same person? (yes/no): ").strip().lower()
                if ans == 'exit':
                    raise AbortEntry()
                if ans == 'yes':
                    return True
                elif ans == 'no':
                    return False
                else:
                    print("\n(!) Invalid input type again ...\n")
        else:
            return False 
            
    return True

def view_previous_data():
    clear()
    print("="*80)
    print("                      PREVIOUS PATIENT RECORDS")
    print("="*80)
    
    if not os.path.exists(DATABASE_FILE) or os.stat(DATABASE_FILE).st_size == 0:
        print("\nno data entered :( \n")
        input("Press [ENTER] to return to the Main Menu...")
        return

    while True:
        df = pd.read_csv(DATABASE_FILE)
        df['Name'] = df['First Name'] + ' ' + df['Last Name']
        cols = [c.lower() for c in df.columns]
        
        print("\n" + "-"*80)
        query = input("Type 'all' or a specific column (id, name, age, bmi...) or type 'exit' to exit to main menu: ").strip().lower()
        
        if query == 'exit':
            return
            
        if query == 'all':
            print("\n--- ALL RECORDS ---")
            print(df.drop(columns=['Name']).to_string(index=False))
            input("\nPress [ENTER] to continue...")
            continue
            
        if query in cols:
            actual_col = df.columns[cols.index(query)]
            
            while True:
                val = input(f"\nSpecify the value for {actual_col} (or type 'exit' to change column): ").strip().lower()
                if val == 'exit':
                    break 
                
                if not val:
                    print("\n(!) Invalid input type again ...\n")
                    continue
                
                if actual_col.lower() == 'date':
                    if re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", val):
                        d, m, y = map(int, val.split('/'))
                        if not (2025 <= y <= 2125) or not (1 <= m <= 12):
                            print("\n(!) Please enter the current date , type again ...\n")
                            continue
                        if m == 2:
                            if y % 400 == 0: max_d = 29
                            elif y % 100 == 0: max_d = 28
                            elif y % 4 == 0: max_d = 29
                            else: max_d = 28
                        elif m in [1, 3, 5, 7, 8, 10, 12]: max_d = 31
                        else: max_d = 30
                        
                        if not (1 <= d <= max_d):
                            print("\n(!) Please enter the current date , type again ...\n")
                            continue
                        val = f"{d:02d}/{m:02d}/{y}"
                    else:
                        print("\n(!) Invalid input type again ...\n")
                        continue
                
                elif actual_col.lower() == 'age':
                    try:
                        num = int(val)
                        if not (1 <= num <= 150):
                            print("\n(!) Wrong age ..type again ...\n")
                            continue
                    except ValueError:
                        print("\n(!) Invalid input type again ...\n")
                        continue
                        
                elif actual_col.lower() in ['pregnancies', 'glucose', 'bmi', 'blood pressure', 'skin thickness', 'insulin', 'dpf', 'confidence(%)']:
                    try:
                        float(val)
                    except ValueError:
                        print("\n(!) Invalid input type again ...\n")
                        continue
                        
                elif actual_col.lower() == 'id':
                    if not val.isdigit():
                        print("\n(!) Invalid input type again ...\n")
                        continue

                # --- BULLETPROOF CASE-INSENSITIVE NAME SEARCH ---
                if actual_col.lower() == 'name':
                    # .str.lower() ensures the database string is treated as lowercase during the check
                    filtered_df = df[(df['Name'].str.lower() == val) | (df['First Name'].str.lower().str.startswith(val))]
                else:
                    filtered_df = df[df[actual_col].astype(str).str.lower() == val]
                
                if filtered_df.empty:
                    print(f"\nNo records found for {actual_col} = {val}.\n")
                    continue
                
                basic_cols = ['ID', 'Date', 'First Name', 'Last Name', 'Gender', 'Glucose', 'BMI', 'Age', 'Result', 'Confidence(%)']
                print("\n--- BASIC DETAILS ---")
                print(filtered_df[basic_cols].to_string(index=False))
                
                while True:
                    full_req = input("\nType 'full' for full details, 'exit' to change column, or [ENTER] to search another value: ").strip().lower()
                    if full_req in ['full', 'exit', '']:
                        break
                    print("\n(!) Invalid input type again ...\n")
                
                if full_req == 'exit':
                    break
                elif full_req == 'full':
                    full_display_df = filtered_df.drop(columns=['Name'])
                    print("\n--- FULL DETAILS ---")
                    print(full_display_df.to_string(index=False))
                    
                    while True:
                        next_action = input("\nPress [ENTER] to search another value or type 'exit' to change column: ").strip().lower()
                        if next_action in ['exit', '']:
                            break
                        print("\n(!) Invalid input type again ...\n")
                        
                    if next_action == 'exit':
                        break
        else:
            print("\n(!) Invalid input type again ...\n")

# --- 4. NEW DATA ENTRY LOGIC ---
def enter_new_patient():
    global SESSION_DATE
    
    try:
        if not SESSION_DATE:
            clear()
            print("="*50)
            print("NEW CLINICAL DATA ENTRY".center(50))
            print("="*50)
            SESSION_DATE = get_date("Enter Date (dd/mm/yyyy): ")

        clear()
        print("="*50)
        print("NEW CLINICAL DATA ENTRY".center(50))
        print(SESSION_DATE.center(50))
        print("="*50)

        print("\n⚠️ Warning: Please put a space between different words in the full name.\n")
        p_name = get_string("Enter Patient Full Name: ", enforce_space=True).lower()
        
        name_parts = p_name.rsplit(' ', 1)
        first_name = name_parts[0]
        last_name = name_parts[1]
        
        while True:
            p_id = get_string("Enter Patient ID: ", enforce_digit=True)
            if check_duplicate_id(p_id, p_name):
                break
            else:
                print("\n(!) Dublicate id type again.....\n")

        while True:
            gender = get_string("Enter Patient Gender (M/F): ").upper()
            if gender in ['M', 'F']:
                break
            print("\n(!) Invalid input type again ...\n")
            
        v8 = get_age("Enter Age: ")

        print(f"\n[+] Clinical Data Entry for {p_name.title()} ({gender}):")
        
        if gender == 'M':
            v1 = 0.0
            print("  > Pregnancies: 0 (Auto-filled for Male)")
        else:
            v1 = get_float("  > Pregnancies: ")

        v2 = get_float("  > Glucose Level (mg/dL): ")
        v3 = get_float("  > Blood Pressure (mm Hg): ")
        v4 = get_float("  > Skin Thickness (mm): ")
        v5 = get_float("  > Insulin Level (mu U/ml): ")
        v6 = get_float("  > Body Mass Index (BMI): ")
        v7 = get_float("  > Diabetes Pedigree Function: ")

        user_data = np.array([[v1, v2, v3, v4, v5, v6, v7, v8]])
        user_data_scaled = scaler.transform(user_data)
        prediction = model.predict(user_data_scaled)
        prob = model.predict_proba(user_data_scaled)[0][1] * 100

        res_text = "POSITIVE" if prediction[0] == 1 else "NEGATIVE"
        conf_val = prob if prediction[0] == 1 else (100 - prob)

        new_record = pd.DataFrame([{
            "ID": p_id, 
            "Date": SESSION_DATE,
            "First Name": first_name, 
            "Last Name": last_name,
            "Gender": gender,
            "Age": v8,
            "Glucose": v2,
            "BMI": v6,
            "Result": res_text, 
            "Confidence(%)": round(conf_val, 2),
            "Pregnancies": v1,
            "Blood Pressure": v3,
            "Skin Thickness": v4,
            "Insulin": v5,
            "DPF": v7
        }])
        
        if not os.path.exists(DATABASE_FILE):
            new_record.to_csv(DATABASE_FILE, index=False)
        else:
            new_record.to_csv(DATABASE_FILE, mode='a', header=False, index=False)

        print("\n" + "="*50)
        print(f"REPORT FOR ID: {p_id} | NAME: {p_name.title()} | GENDER: {gender}")
        print("-" * 50)
        print(f"RESULT     : {res_text} INDICATIONS")
        print(f"CONFIDENCE : {conf_val:.2f}% Probability")
        print("=" * 50)
        
        while True:
            ans = input("\n[Data Saved] Press [ENTER] without typing anything to start the next person entry, or type 'exit' to go back to main menu: ").strip().lower()
            if ans in ['exit', '']:
                break
            print("\n(!) Invalid input type again ...\n")
            
        if ans == 'exit':
            return 'EXIT'
        else:
            return 'RESTART'
            
    except AbortEntry:
        clear()
        print("Data entry cancelled.")
        while True:
            ans = input("Press [ENTER] to continue taking another person data entry, or type 'exit' to exit to main menu: ").strip().lower()
            if ans in ['exit', '']:
                break
            print("\n(!) Invalid input type again ...\n")
            
        if ans == 'exit':
            return 'EXIT'
        else:
            return 'RESTART'

# --- 5. MAIN MENU LOOP ---
while True:
    clear()
    print("="*50)
    print("      DIABETES RISK ASSESSMENT SYSTEM")
    print("="*50)
    print("  [ENTER] - Press Enter to input new patient data")
    print("  'entry' - Type 'entry' to view previous data")
    print("  'exit'  - Type 'exit' to quit the application")
    print("="*50)
    
    choice = input("\nSelect an option: ").strip().lower()
    
    if choice == 'exit':
        clear()
        print("\nthank u sir please give good marks :)\n")
        break
    elif choice == 'entry':
        view_previous_data()
    elif choice == '':
        while True:
            action = enter_new_patient()
            if action == 'EXIT':
                SESSION_DATE = None 
                break 
            elif action == 'RESTART':
                continue 
    else:
        print("\n(!) Invalid input type again ...\n")
        time.sleep(1.5)
