import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

    
def analyze_6430(df, model, X=None, y=None):
    print(f"Dataset shape: {df.shape}")
    print(f"Columns in dataset: {list(df.columns)}")
    
    # Define target and features
    target_col = 'member of a labor union'
    feature_cols = [
        'capital losses',
        'age',
        'region of previous residence',
        'state of previous residence',
        'detailed occupation recode',
        "fill inc questionnaire for veteran's admin",
        'migration code-change in msa',
        'country of birth self'
    ]
    
    print(f"\nTarget column: {target_col}")
    print(f"Feature columns: {feature_cols}")
    
    # Check if all columns exist
    missing_cols = [col for col in [target_col] + feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        exit(1)
    
    # Basic statistics
    print(f"\nTarget variable analysis:")
    print(f"  Data type: {df[target_col].dtype}")
    print(f"  Unique values: {df[target_col].nunique()}")
    print(f"  Value counts:")
    print(df[target_col].value_counts())
    
    print(f"\nFeature variable analysis:")
    for feature in feature_cols:
        print(f"\n{feature}:")
        print(f"  Data type: {df[feature].dtype}")
        print(f"  Missing values: {df[feature].isnull().sum()}")
        
        if df[feature].dtype in ['object', 'category']:
            print(f"  Unique values: {df[feature].nunique()}")
            print(f"  Top 5 values:")
            top_values = df[feature].value_counts().head()
            for val, count in top_values.items():
                pct = (count / len(df)) * 100
                print(f"    {val}: {count} ({pct:.1f}%)")
        else:
            print(f"  Min: {df[feature].min():.2f}")
            print(f"  Max: {df[feature].max():.2f}")
            print(f"  Mean: {df[feature].mean():.2f}")
            print(f"  Median: {df[feature].median():.2f}")
    
    if X is None:
        # Prepare data for modeling
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values
        initial_rows = len(df)
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]
        print(f"\nRows after removing missing values: {len(X)} (dropped {initial_rows - len(X)} rows)")
        
        if len(X) == 0:
            print("No data remaining after removing missing values!")
            exit(1)
        
        # Encode categorical variables
        label_encoders = {}
        for col in feature_cols:
            if X[col].dtype in ['object', 'category']:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
                print(f"Encoded categorical column: {col}")
        
        # Encode target variable if categorical
        if y.dtype in ['object', 'category']:
            target_encoder = LabelEncoder()
            y_encoded = target_encoder.fit_transform(y)
            print(f"Encoded categorical target column")
            print(f"Target classes: {target_encoder.classes_}")
        else:
            y_encoded = y
            target_encoder = None
        
    print(f"\nFinal dataset statistics:")
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Target vector shape: {y.shape}")
    print(f"  Target distribution:")
    if target_encoder:
        target_counts = pd.Series(y_encoded).value_counts().sort_index()
        for encoded_val, count in target_counts.items():
            original_val = target_encoder.classes_[encoded_val]
            pct = (count / len(y_encoded)) * 100
            print(f"    {original_val} (encoded as {encoded_val}): {count} ({pct:.1f}%)")
    else:
        target_counts = y.value_counts().sort_index()
        for val, count in target_counts.items():
            pct = (count / len(y)) * 100
            print(f"    {val}: {count} ({pct:.1f}%)")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train the model
    #model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    print(f"\nModel Parameters:")
    for param, value in model.get_params().items():
        print(f"  {param}: {value}")
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate prediction confidence (max probability for each prediction)
    prediction_confidence = np.max(y_pred_proba, axis=1)
    
    # Calculate standard metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    precision_micro = precision_score(y_test, y_pred, average='micro')
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    
    recall_macro = recall_score(y_test, y_pred, average='macro')
    recall_micro = recall_score(y_test, y_pred, average='micro')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    
    print(f"\nMODEL PERFORMANCE:")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Precision (micro): {precision_micro:.4f}")
    print(f"Precision (weighted): {precision_weighted:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")
    print(f"Recall (micro): {recall_micro:.4f}")
    print(f"Recall (weighted): {recall_weighted:.4f}")
    
    # Confidence statistics
    print(f"\nConfidence Statistics:")
    print(f"  Mean confidence: {prediction_confidence.mean():.4f}")
    print(f"  Median confidence: {np.median(prediction_confidence):.4f}")
    print(f"  Min confidence: {prediction_confidence.min():.4f}")
    print(f"  Max confidence: {prediction_confidence.max():.4f}")
    
    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    if target_encoder:
        print(classification_report(y_test, y_pred, target_names=[str(cls) for cls in target_encoder.classes_]))
    else:
        print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance:")
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Model analysis
    print(f"\nModel Analysis:")
    train_accuracy = model.score(X_train, y_train)
    print(f"  Training accuracy: {train_accuracy:.4f}")
    print(f"  Test accuracy: {accuracy:.4f}")
    print(f"  Overfitting gap: {train_accuracy - accuracy:.4f}")
    
    # Precision by confidence percentiles
    print(f"\nPRECISION BY CONFIDENCE PERCENTILES:")
    print("="*60)
    
    # Create confidence analysis dataframe
    confidence_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_pred,
        'confidence': prediction_confidence,
        'correct': y_test == y_pred
    })
    
    # Sort by confidence (descending)
    confidence_df_sorted = confidence_df.sort_values('confidence', ascending=False)
    
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    for percentile in percentiles:
        num_rows = int(len(confidence_df_sorted) * percentile / 100)
        top_rows = confidence_df_sorted.head(num_rows)
        
        correct_predictions = top_rows['correct'].sum()
        total_predictions = len(top_rows)
        precision = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        min_confidence = top_rows['confidence'].min()
        max_confidence = top_rows['confidence'].max()
        
        print(f"Top {percentile:3d}%:")
        print(f"  Samples: {total_predictions:,}")
        print(f"  Correct: {correct_predictions:,}")
        print(f"  Precision: {precision:.4f} ({precision*100:.1f}%)")
        print(f"  Confidence range: {min_confidence:.4f} - {max_confidence:.4f}")
        print()
    
    # Sample predictions
    print(f"\nSample Predictions (first 10):")
    for i in range(min(10, len(X_test))):
        true_val = y_test[i]  # Changed from y_test.iloc[i] to y_test[i]
        pred_val = y_pred[i]
        conf = prediction_confidence[i]
        
        if target_encoder:
            true_label = target_encoder.classes_[true_val]
            pred_label = target_encoder.classes_[pred_val]
        else:
            true_label = true_val
            pred_label = pred_val
        
        status = 'CORRECT' if true_val == pred_val else 'WRONG'
        print(f"  True: {true_label} | Pred: {pred_label} | Conf: {conf:.3f} | {status}")
    
    print(f"\n=== ANALYZE_6430 END ===")