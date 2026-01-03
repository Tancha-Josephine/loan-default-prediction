# Loan Default Prediction Interface
# Run this script after training your model in the Jupyter notebook

import pickle
import pandas as pd
import numpy as np
from colorama import init, Fore, Back, Style
import warnings
warnings.filterwarnings('ignore')

# Initialize colorama for colored terminal output
init(autoreset=True)

# =============================================================================
# LOAD TRAINED MODEL AND PREPROCESSING OBJECTS
# =============================================================================

def load_model_artifacts():
    """Load the trained model, scaler, and label encoders"""
    try:
        with open('gradient_boosting_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        print(Fore.GREEN + "✅ Model and preprocessing objects loaded successfully!\n")
        return model, scaler, label_encoders
    
    except FileNotFoundError as e:
        print(Fore.RED + "❌ Error: Model files not found!")
        print(Fore.YELLOW + "Please run the Jupyter notebook first to train and save the model.")
        return None, None, None

# =============================================================================
# PREDICTION FUNCTION
# =============================================================================

def predict_loan_default(model, input_data):
    """
    Predict loan default probability
    
    Parameters:
    -----------
    model : trained Gradient Boosting model
    input_data : DataFrame with loan features
    
    Returns:
    --------
    dict with prediction results
    """
    try:
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        return {
            'prediction': prediction,
            'prediction_label': 'DEFAULT' if prediction == 1 else 'NON-DEFAULT',
            'default_probability': probability[1] * 100,
            'non_default_probability': probability[0] * 100
        }
    except Exception as e:
        print(Fore.RED + f"❌ Prediction Error: {str(e)}")
        return None

# =============================================================================
# DISPLAY RESULTS
# =============================================================================

def display_prediction(result, input_data):
    """Display prediction results in a formatted way"""
    
    print("\n" + "="*80)
    print(Fore.CYAN + Style.BRIGHT + "🎯 LOAN DEFAULT PREDICTION RESULTS")
    print("="*80 + "\n")
    
    # Display input summary
    print(Fore.YELLOW + "📋 INPUT SUMMARY:")
    print("-" * 80)
    for key, value in input_data.items():
        print(f"   {key:.<35} {value}")
    
    print("\n" + "-" * 80)
    
    # Display prediction
    if result['prediction'] == 1:
        print(Fore.RED + Style.BRIGHT + "\n⚠️  PREDICTION: DEFAULT RISK DETECTED")
        print(Fore.RED + f"   Default Probability: {result['default_probability']:.2f}%")
        print(Fore.GREEN + f"   Non-Default Probability: {result['non_default_probability']:.2f}%")
        
        # Risk level
        if result['default_probability'] > 70:
            risk_level = "🔴 VERY HIGH RISK"
            color = Fore.RED
        elif result['default_probability'] > 50:
            risk_level = "🟠 HIGH RISK"
            color = Fore.YELLOW
        else:
            risk_level = "🟡 MODERATE RISK"
            color = Fore.YELLOW
        
        print(color + f"\n   Risk Level: {risk_level}")
        print(Fore.RED + "\n   ⛔ Recommendation: LOAN NOT RECOMMENDED")
        
    else:
        print(Fore.GREEN + Style.BRIGHT + "\n✅ PREDICTION: LOW DEFAULT RISK")
        print(Fore.GREEN + f"   Non-Default Probability: {result['non_default_probability']:.2f}%")
        print(Fore.RED + f"   Default Probability: {result['default_probability']:.2f}%")
        
        # Risk level
        if result['default_probability'] < 10:
            risk_level = "🟢 VERY LOW RISK"
            color = Fore.GREEN
        elif result['default_probability'] < 20:
            risk_level = "🟢 LOW RISK"
            color = Fore.GREEN
        else:
            risk_level = "🟡 MODERATE RISK"
            color = Fore.YELLOW
        
        print(color + f"\n   Risk Level: {risk_level}")
        print(Fore.GREEN + "\n   ✅ Recommendation: LOAN APPROVED")
    
    print("\n" + "="*80 + "\n")

# =============================================================================
# GET USER INPUT
# =============================================================================

def get_user_input(label_encoders):
    """Interactive function to get loan application details from user"""
    
    print(Fore.CYAN + Style.BRIGHT + "\n🏦 LOAN APPLICATION - ENTER BORROWER DETAILS")
    print("="*80 + "\n")
    
    try:
        # Numerical inputs
        print(Fore.YELLOW + "💰 Financial Information:")
        annual_income = float(input("   Annual Income ($): "))
        debt_to_income = float(input("   Debt-to-Income Ratio (%): "))
        loan_amount = float(input("   Loan Amount Requested ($): "))
        interest_rate = float(input("   Interest Rate (%): "))
        installment = float(input("   Monthly Installment ($): "))
        
        print(Fore.YELLOW + "\n📊 Credit Information:")
        emp_length = float(input("   Employment Length (years, 0-10): "))
        total_credit_lines = int(input("   Total Credit Lines: "))
        open_credit_lines = int(input("   Open Credit Lines: "))
        delinq_2y = int(input("   Delinquencies in Last 2 Years: "))
        inquiries_last_12m = int(input("   Credit Inquiries Last 12 Months: "))
        public_record_bankrupt = int(input("   Bankruptcy Records (0 or 1): "))
        
        print(Fore.YELLOW + "\n🏠 Loan Details:")
        term = int(input("   Loan Term (36 or 60 months): "))
        
        # Categorical inputs
        print(Fore.YELLOW + "\n📝 Categorical Information:")
        print("   Homeownership Options: RENT, OWN, MORTGAGE")
        homeownership = input("   Homeownership Status: ").upper()
        
        print("   Credit Grade Options: A, B, C, D, E, F, G")
        grade = input("   Credit Grade: ").upper()
        
        print("   Verified Income Options: Verified, Source Verified, Not Verified")
        verified_income = input("   Income Verification Status: ")
        
        # Encode categorical variables
        homeownership_encoded = label_encoders['homeownership'].transform([homeownership])[0]
        grade_encoded = label_encoders['grade'].transform([grade])[0]
        verified_income_encoded = label_encoders['verified_income'].transform([verified_income])[0]
        
        # Create input dictionary
        input_data = {
            'annual_income': annual_income,
            'debt_to_income': debt_to_income,
            'interest_rate': interest_rate,
            'loan_amount': loan_amount,
            'emp_length': emp_length,
            'term': term,
            'homeownership': homeownership_encoded,
            'grade': grade_encoded,
            'installment': installment,
            'verified_income': verified_income_encoded,
            'total_credit_lines': total_credit_lines,
            'open_credit_lines': open_credit_lines,
            'delinq_2y': delinq_2y,
            'inquiries_last_12m': inquiries_last_12m,
            'public_record_bankrupt': public_record_bankrupt
        }
        
        # Create display dictionary (with readable values)
        display_data = {
            'Annual Income': f"${annual_income:,.2f}",
            'Debt-to-Income Ratio': f"{debt_to_income:.2f}%",
            'Loan Amount': f"${loan_amount:,.2f}",
            'Interest Rate': f"{interest_rate:.2f}%",
            'Monthly Installment': f"${installment:,.2f}",
            'Employment Length': f"{emp_length:.1f} years",
            'Loan Term': f"{term} months",
            'Homeownership': homeownership,
            'Credit Grade': grade,
            'Income Verification': verified_income,
            'Total Credit Lines': total_credit_lines,
            'Open Credit Lines': open_credit_lines,
            'Delinquencies (2y)': delinq_2y,
            'Credit Inquiries (12m)': inquiries_last_12m,
            'Bankruptcy Records': public_record_bankrupt
        }
        
        return input_data, display_data
    
    except ValueError as e:
        print(Fore.RED + f"\n❌ Invalid input: {str(e)}")
        print(Fore.YELLOW + "Please enter valid numerical values.")
        return None, None
    except Exception as e:
        print(Fore.RED + f"\n❌ Error: {str(e)}")
        return None, None

# =============================================================================
# BATCH PREDICTION FROM CSV
# =============================================================================

def batch_prediction(model, label_encoders, csv_file):
    """Make predictions for multiple loan applications from CSV file"""
    
    try:
        print(Fore.CYAN + f"\n📂 Loading data from {csv_file}...")
        df = pd.read_csv(csv_file)
        
        print(Fore.GREEN + f"✅ Loaded {len(df)} loan applications")
        
        # Make predictions
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]
        
        # Add results to dataframe
        df['Prediction'] = ['DEFAULT' if p == 1 else 'NON-DEFAULT' for p in predictions]
        df['Default_Probability'] = probabilities * 100
        
        # Save results
        output_file = 'batch_predictions.csv'
        df.to_csv(output_file, index=False)
        
        print(Fore.GREEN + f"✅ Predictions saved to {output_file}")
        
        # Summary
        default_count = sum(predictions)
        print(Fore.YELLOW + f"\n📊 Summary:")
        print(f"   Total Applications: {len(predictions)}")
        print(f"   Predicted Defaults: {default_count} ({default_count/len(predictions)*100:.1f}%)")
        print(f"   Predicted Non-Defaults: {len(predictions)-default_count} ({(len(predictions)-default_count)/len(predictions)*100:.1f}%)")
        
        return df
        
    except FileNotFoundError:
        print(Fore.RED + f"❌ Error: File '{csv_file}' not found!")
        return None
    except Exception as e:
        print(Fore.RED + f"❌ Error: {str(e)}")
        return None

# =============================================================================
# QUICK PREDICTION WITH SAMPLE DATA
# =============================================================================

def quick_predict_sample(model, label_encoders):
    """Make a quick prediction with predefined sample data"""
    
    print(Fore.CYAN + "\n🚀 QUICK PREDICTION WITH SAMPLE DATA\n")
    
    # Sample borrower profiles
    samples = [
        {
            'name': 'Low Risk Borrower',
            'data': {
                'annual_income': 75000,
                'debt_to_income': 10.5,
                'interest_rate': 8.5,
                'loan_amount': 12000,
                'emp_length': 8.0,
                'term': 36,
                'homeownership': 'OWN',
                'grade': 'A',
                'installment': 377.0,
                'verified_income': 'Verified',
                'total_credit_lines': 25,
                'open_credit_lines': 12,
                'delinq_2y': 0,
                'inquiries_last_12m': 1,
                'public_record_bankrupt': 0
            }
        },
        {
            'name': 'High Risk Borrower',
            'data': {
                'annual_income': 28000,
                'debt_to_income': 35.8,
                'interest_rate': 22.5,
                'loan_amount': 18000,
                'emp_length': 1.0,
                'term': 60,
                'homeownership': 'RENT',
                'grade': 'F',
                'installment': 512.0,
                'verified_income': 'Not Verified',
                'total_credit_lines': 8,
                'open_credit_lines': 6,
                'delinq_2y': 3,
                'inquiries_last_12m': 5,
                'public_record_bankrupt': 1
            }
        }
    ]
    
    for idx, sample in enumerate(samples, 1):
        print(Fore.YELLOW + f"\n{'='*80}")
        print(Fore.YELLOW + f"Sample {idx}: {sample['name']}")
        print(Fore.YELLOW + f"{'='*80}")
        
        # Encode categorical variables
        data = sample['data'].copy()
        data['homeownership'] = label_encoders['homeownership'].transform([data['homeownership']])[0]
        data['grade'] = label_encoders['grade'].transform([data['grade']])[0]
        data['verified_income'] = label_encoders['verified_income'].transform([data['verified_income']])[0]
        
        # Create DataFrame
        input_df = pd.DataFrame([data])
        
        # Make prediction
        result = predict_loan_default(model, input_df)
        
        if result:
            display_prediction(result, sample['data'])

# =============================================================================
# MAIN MENU
# =============================================================================

def main():
    """Main function with menu interface"""
    
    print(Fore.CYAN + Style.BRIGHT + "\n" + "="*80)
    print("🏦 LOAN DEFAULT PREDICTION SYSTEM")
    print("="*80)
    
    # Load model
    model, scaler, label_encoders = load_model_artifacts()
    
    if model is None:
        return
    
    while True:
        print(Fore.CYAN + "\n" + "="*80)
        print("MAIN MENU")
        print("="*80)
        print(Fore.WHITE + """
        1. 🎯 Make Single Prediction (Manual Input)
        2. 🚀 Quick Prediction with Sample Data
        3. 📊 Batch Prediction from CSV
        4. 🚪 Exit
        """)
        
        choice = input(Fore.YELLOW + "Select an option (1-4): ").strip()
        
        if choice == '1':
            # Single prediction
            input_data, display_data = get_user_input(label_encoders)
            
            if input_data is not None:
                # Create DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Make prediction
                result = predict_loan_default(model, input_df)
                
                if result:
                    display_prediction(result, display_data)
        
        elif choice == '2':
            # Quick sample prediction
            quick_predict_sample(model, label_encoders)
        
        elif choice == '3':
            # Batch prediction
            csv_file = input(Fore.YELLOW + "\nEnter CSV filename (with features): ").strip()
            batch_prediction(model, label_encoders, csv_file)
        
        elif choice == '4':
            # Exit
            print(Fore.GREEN + "\n👋 Thank you for using Loan Default Prediction System!")
            print("="*80 + "\n")
            break
        
        else:
            print(Fore.RED + "\n❌ Invalid choice. Please select 1-4.")
        
        # Ask if user wants to continue
        if choice in ['1', '2', '3']:
            continue_choice = input(Fore.YELLOW + "\nMake another prediction? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print(Fore.GREEN + "\n👋 Thank you for using Loan Default Prediction System!")
                print("="*80 + "\n")
                break

# =============================================================================
# RUN THE APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()