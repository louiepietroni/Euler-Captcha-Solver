import requests
import shutil

url = 'https://projecteuler.net/captcha/show_captcha.php'

start_index = 300
number_to_generate = 50

for index in range(start_index, start_index + number_to_generate):
    file_name = 'Captchas/captcha%d.png' % index

    res = requests.get(url, stream=True)

    if res.status_code == 200:
        with open(file_name, 'wb') as f:
            shutil.copyfileobj(res.raw, f)
        print('Image successfully Downloaded: ', file_name)
    else:
        print('Image Couldn\'t be retrieved')
