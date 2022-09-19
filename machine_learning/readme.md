## 資料來源:  
>[ 參考教學影片 ](https://l.facebook.com/l.php?u=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DpqNCD_5r0IU%26t%3D2541s&h=AT1YTcr7H6_Uu2LU9naGenvq0kplgOsO4P8Eciijk7Xf4YakLBppeuKEXuV-F40rDBf8eV_1zWPkVoLcCU_CLOnHXawSDvXHFYV_Bbba9PW1eCtdroEJY38UKRCchpq3UF1jb99LpdIdbRk&s=1/"Title")  

| buying | maint | doors | persons | lug_boot | safety | class|
| :-----:| :----: | :----: |:----:| :----:| :----: | :----:|
| vhigh | vhigh | 2 | 2 | small | low | unacc |
| high | high | 3 | 4 | med | med | acc |
| med | med | 4 | more | big | high | good|
| low | low | 5 more | |||vgood |
## knn.py的主要功能是依據:
>[ 數據來源 ](https://archive.ics.uci.edu/ml/datasets/car+evaluation)中提供的數據中用` buying ` , ` maint ` , ` doors ` , ` persons ` , ` lug_boot ` , `safty`來訓練模型判斷` class `，而` class `是接受度高不高，也就是這種型態的車受不受歡迎。
## 筆記:  
> 在knn.py中:  
>1.調整X中的訓練取樣數據會影響最終的準確度  
>2.調整` knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform') `的參數也會影響最終的準確度
## knn公式:  

> $ d(x,y) = \sqrt{\sum_{i=1}^{n}(y_i-x_i)^{2}  } $
