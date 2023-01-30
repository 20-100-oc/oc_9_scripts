# Front end app to interact with an Azure function.
# To start app, type in the command prompt: streamlit run <name_of_this_file>

import streamlit as st
import requests



def main():
    #url = 'http://localhost:7071/api/HttpTrigger2' # local url
    url = 'https://oc-9-func-2.azurewebsites.net/api/HttpTrigger2'

    st.title('Article Recommendation')
    user_id = st.number_input(label='Insert user ID here:', min_value=0)

    predict_btn = st.button('Get recommendations')
    if predict_btn:
        try:
            keys = {'userID': str(user_id)}
            raw_res = requests.get(url, params=keys)
            res = eval(raw_res.text)

            received_user_id = int(res['user_id'])
            st.write('User ID:', received_user_id)
            assert received_user_id == user_id, '"received_user_id" is not "like user_id"'

            received_user_recs = res['user_recs']
            st.write('Recommendations list:', received_user_recs)

        except Exception as e:
            st.write(raw_res)
            st.write('Raw json returned by api:')
            st.write(raw_res.text)
            raise e



if __name__ == '__main__':
    main()
