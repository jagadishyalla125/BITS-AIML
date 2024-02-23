# ***********************************************************************************************
# Title: Airline Customer Satisfaction Prediction
# Subject: Capstone Project - PCAMZC321
#
# Mentor:
# Mr. Satyaki Dasgupta
#
# Student Names :
# Robin Mathew       | 2021AIML003
# Jagadish Yalla     | 2021AIML064
# Abhishek Agarwal   | 2021AIML045
# Deepa Krishnaswami | 2021AIML049
# ***********************************************************************************************
#
# Steps to start this GUI
# 1) Pre-requisite : Make sure to install streamlit python module using "pip install streamlit"
# 2) Open anaconda command prompt or anaconda power shell
# 3) Change the current working directory to the path where this py file is located
# 4) Launch the GUI using "streamlit run AirlineCustomerSatisfaction_Deployment.py" command
# ***********************************************************************************************


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import streamlit as st
import sys
#sys.path.append('C:/Users/robin/OneDrive/Documents/PGCP AIML/CapstoneProject/ProjectFiles')

from AirlineCustomerSatisfaction_Functions import dropColumns, renameAllColumns, performLabelEncoding, scaleData, performMinMaxScaling, DBScanOutlier, performBinning, createBin
from os.path import exists
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def DisplayStatistics(data):
	st.write("Sample Data:")
	st.write(data.head())
	st.write("Shape:", data.shape)
	#st.write(data.shape)
	st.write("Missing Values:", data.isnull().values.any())
	#st.write(data.isnull().values.any())
	st.write("Duplicates:", data.duplicated().values.any())
	#st.write(data.duplicated().values.any())
	st.write("Key Stats:")
	st.write(data.describe())

def OnFileUpload(uploaded_file):#, org_cols):
	with st.spinner('Reading the data...'):
		data = pd.read_csv(uploaded_file)# , index_col=False, names=org_cols)
	st.success('Data uploaded Successfully. Basic Info:')
	return data

def OnPreProcessing(actions, data, colsToDrop, renamedCols, colsToEncode, targetColumn, columnsToBin):
    with st.spinner('Please wait...'):

        if ( "Drop Unwanted Columns" in actions ):           
            data = dropColumns(data, colsToDrop)
            st.success('Dropped Columns: '+str(colsToDrop)+' Successfully')
        if ( "Rename Columns" in actions ):
            data = renameAllColumns(data, renamedCols)
            st.success('Renamed all columns Successfully to: '+str(renamedCols))
        if ( "Remove Duplicates" in actions ):
            data = data.drop_duplicates()
            st.success('Dropped duplicate data Successfully')                
        if ( "Handle Missing Values" in actions ):
            data = data.dropna()
            st.success('Dropped missing records Successfully')                
        if ( "Encode" in actions ):
            data = performLabelEncoding(data, colsToEncode)
            st.success('Label Encoded the columns: '+str(colsToEncode)+' Successfully')
        if ( "Remove Outliers" in actions ):
            outlierRows=DBScanOutlier(data)
            data.drop(outlierRows, axis=0, inplace=True)
            st.success('Removed outliers Successfully')
        if("Binning" in actions):
            data=performBinning(data, columnsToBin)
            st.success('Binning done on dataset')

        if ( "Scale" in actions ):
            data = scaleData(data, targetColumn)
            st.success('Scaled data Successfully using Min-Max Scaler')

            
            
    return data


def OnPredict(dataSet, modelName, modelFile, data, targetColumn, columnsToBin):
    modellingData=data.copy()

    if ( exists(modelFile) != True ) :
        st.error("The model {} is not available. Try a different model!".format(modelFile))
        return

    with st.spinner('Please wait...'):
        model = pickle.load(open(modelFile, 'rb'))
        xtest = modellingData.drop(columns=targetColumn)
        ytest = modellingData[targetColumn]
        ypred = model.predict(xtest)

        st.success('Prediction Completed Successfully with Model: '+str(modelName)+' and Dataset: '+str(dataSet))
        st.write('<b>Accuracy: </b>'f"{100*accuracy_score(ytest, ypred):.2f}%", unsafe_allow_html=True)
        #st.write()

        st.write('<b>Classification Report: </b>', unsafe_allow_html=True)
        cr = classification_report(ytest, ypred, output_dict=True)
        df = pd.DataFrame(cr).transpose()
        st.write(df.head(7))

        st.write("<b>Confusion Matrix: </b>", unsafe_allow_html=True)
        cm = confusion_matrix(ytest, ypred)
        st.write(cm)
        st.write("<b>Confusion Matrix - Graphical View: </b>", unsafe_allow_html=True)

        f, ax = plt.subplots(figsize=(15,15))
        sns.set(font_scale=1.25)
        st.write(sns.heatmap(cm, square=True, annot=True, fmt="d", cmap="RdYlGn"))
        st.pyplot(f)

        f, ax = plt.subplots(figsize=(15,15))
        sns.set(font_scale=1.25)
        st.write(sns.heatmap(cm/np.sum(cm), square=True, annot=True, fmt='.2%', cmap='RdYlGn'))
        st.pyplot(f)
        st.balloons()

    return


def main():
    
    targetColumn='Satisfaction'
    dataSetList=['UnBinned','Binned']
    columnsToBin=['Age', 'Flight_Distance', 'Departure_Delay']    
    model_list=['Light Gradient Boosting','Gradient Boosting','XGB','Ada Boost','Stacking Classifier', 'CatBoost Classifier']
    model_dict = { 'Light Gradient Boosting':'LightGradientBoosting.sav', 'Gradient Boosting':'Gradient Boosting.sav', 'XGB':'xgb.sav', 'Ada Boost':'ada boost.sav', 'Stacking Classifier':'stacking.sav', 'CatBoost Classifier':'CatBoostClassifier.sav'}
    actions_list = ['Drop Unwanted Columns', 'Rename Columns', 'Remove Duplicates', 'Handle Missing Values', 'Encode', 'Remove Outliers', 'Scale']	
    colsToDrop = ['Unnamed: 0', 'id','Arrival Delay in Minutes']
    colsToEncode = ['Gender','Customer_Type','Type_Of_Travel','Class', targetColumn]
    renamedCols = ['Gender','Customer_Type','Age','Type_Of_Travel','Class','Flight_Distance','Inflight_Wifi_Service','Departure_Arrival_Time_Convenience','Ease_Of_Online_Booking','Gate_Location','Food_And_Drink','Online_Boarding','Seat_Comfort','Inflight_Entertainment','Onboard_Service','Legroom_Service','Baggage_Handling','Checkin_Service','Inflight_Service','Cleanliness','Departure_Delay','Satisfaction']
    st.markdown('<h5 style=text-align:center>PGCP in AI & ML - Cohort 6 | Capstone Project - PCAMZC321</h5>', unsafe_allow_html=True)
    st.markdown('<h6 style=text-align:center>Group - 1 | Robin Mathew | Jagadish Yalla | Abhishek Agarwal | Deepa Krishnaswami</h6>', unsafe_allow_html=True)
    st.image('Deploy.jpg', width=700)
    
    
    hide_streamlit_style = """
    
    <style>
    MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    
    
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 300px;
    }
    
    </style>

    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    # Sidebar - Header
    st.sidebar.header("Menu")
    
    with st.sidebar.expander('Read Me'):
        tab1, tab2, tab3 = st.tabs(["Pre-Processing", "DataSets", "Models"])
        with tab2:
            st.markdown('UnBinned')
            st.markdown('Binned')
            
        with tab3:
            st.markdown('Light Gradient Boosting')
            st.markdown('Gradient Boosting')
            st.markdown('XGB')
            st.markdown('Ada Boost')
            st.markdown('Stacking Classifier')
            st.markdown('CatBoost Classifier (Binned)')
        with tab1:
            st.markdown('Drop Unwanted Columns')
            st.markdown('Rename Columns')
            st.markdown('Remove Duplicates (Optional)')
            st.markdown('Handle Missing Values (Optional)')
            st.markdown('Encode')
            st.markdown('Remove Outliers (Optional)')
            st.markdown('Scale')


    # Sidebar - Upload Data File
    with st.sidebar.expander('Upload File'):
        uploaded_file = st.file_uploader("Upload .csv File", type='csv')
    if ( uploaded_file is not None ):
        data = OnFileUpload(uploaded_file)
        DisplayStatistics((data))
    else: return
    

    # Sidebar -  Pre-processing multi-select widget & button
    preprocessingButtonPressed=False
    with st.sidebar.expander('Pre-Processing'):
        preprocess_actions = st.multiselect('Choose Pre-Processing Options', actions_list, [actions_list[0],actions_list[1],actions_list[4],actions_list[6]])
        if ( len(preprocess_actions) ):
            if st.button('Perform Pre-Processing'):
                preprocessingButtonPressed=True
    if(preprocessingButtonPressed):
        data = OnPreProcessing(preprocess_actions, data, colsToDrop, renamedCols, colsToEncode, targetColumn, columnsToBin)
        DisplayStatistics((data))

    preprocessingButtonPressed=False
    with st.sidebar.expander('Prediction'):
	# Sidebar -  Model & Dataset Select widgets & Predict button
        dataSet = st.selectbox('Choose Dataset', dataSetList)
        modelName = st.selectbox('Choose Model', model_list)

        if ( ( modelName is not None ) and ( dataSet is not None ) ):
            if st.button('Predict'):
                preprocessingButtonPressed=True
    if(preprocessingButtonPressed):
        if(modelName=='CatBoost Classifier'):
            dataSet='Binned'
        modelFile = dataSet+'_'+model_dict.get(modelName)
        
        if(dataSet=='Binned'):
            preprocess_actions.append('Binning')
        
        data = OnPreProcessing(preprocess_actions, data, colsToDrop, renamedCols, colsToEncode, targetColumn, columnsToBin)

        OnPredict(dataSet, modelName, modelFile, data, targetColumn, columnsToBin)


if __name__ == '__main__':
    main()

    # end of file