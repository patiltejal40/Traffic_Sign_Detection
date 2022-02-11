import numpy as np
import streamlit as st
import cv2 as cv
from PIL import Image
from keras.models import load_model





# Label traffic signs
labels_dict = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing vehicle over 3.5 tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicle > 3.5 tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End speed + passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing vehicle > 3.5 tons'
}


@st.cache
def sign_predict(image):
    model = load_model('Model/Traffic_Sign_Classifier_CNN.hdf5')
    image = np.array(image, dtype=np.float32)
    image = image/255
    image = np.reshape(image, (1, 32, 32))
    x = image.astype(np.float32)
    prediction = model.predict(x)
    prediction_max = np.argmax(prediction)
    prediction_label = labels_dict[prediction_max]
    confidence = np.max(prediction)
    return prediction_label, confidence


def main():
    # Set page config and markdowns
    st.set_page_config(page_title='Traffic Signs Classification', page_icon=':car:')
    st.title('Traffic Signs Classification')
    
  
    st.image('./Test Random Images/dash.jpg', use_column_width=True)
    image_usr = st.file_uploader('Upload a photo of traffic sign here', type=['jpg', 'jpeg', 'png'])

    if image_usr is not None:
        col1, col2 = st.columns(2)
        col1.markdown('#### Your picture')
        col2.markdown('#### Your picture 32x32 gray')
        image = Image.open(image_usr)
        with col1:
            st.image(image, use_column_width=True)

        image_np = np.array(image.convert('RGB'))
        image_col = cv.cvtColor(image_np, 1)
        image_gray = cv.cvtColor(image_col, cv.COLOR_BGR2GRAY)
        image_32 = cv.resize(image_gray, (32, 32))
        with col2:
            st.image(image_32, use_column_width=True)

        # Make prediction
        prediction_label, confidence = sign_predict(image_32)

        st.write('##### Prediction:', prediction_label)
        st.write('##### Confidence:', str(confidence))
        st.markdown('***')
    with st.expander("Classes Present in GTSRB dataset"):
        st.write(list(labels_dict.values()))

 

if __name__ == '__main__':
    main()
