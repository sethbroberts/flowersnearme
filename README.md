# flowersnearme

Simple demo of flower classifier built with [fastai](https://www.fast.ai/).

See deployed app on heroku [here](https://vabirds.herokuapp.com/).

This app is built with [streamlit](https://www.streamlit.io/).

To deploy on heroku, I used this [tutorial](https://www.youtube.com/watch?v=skpiLtEN3yk). To ensure the slug size did not exceed heroku's limit, I modified requirements.txt:

```
torch==1.6.0
torchvision==0.7.0
```

changed to

```
--find-links https://download.pytorch.org/whl/torch_stable.html
torch==1.6.0+cpu
torchvision==0.7.0+cpu
```

