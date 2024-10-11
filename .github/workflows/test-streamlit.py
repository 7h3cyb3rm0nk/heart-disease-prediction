import requests
try:
    response = requests.get('http://localhost:8501')
    if response.status_code == 200:
        print('Streamlit app is running successfully!')
    else:
        print('Streamlit app failed to start!')
        exit(1)
except Exception as e:
        print(f'Error: {e}')
        exit(1)
