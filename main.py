import tensorflow as tf
import matplotlib.pyplot as plt #이미지 보여주기
import numpy as np

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data() #의류 이미지 데이터셋
#(인풋 데이터, 정답) (테스트용 데이터...)

#plt.imshow(trainX[1])
#plt.show()

trainX = trainX / 255.0 #이미지 데이터 전처리 0 ~ 255 -> 0 ~ 1로 압축 해서 넣음(선택 사항)
testX = testX / 255.0

trainX = trainX.reshape((trainX.shape[0], 28, 28, 1)) #이미지 전처리 4차원화
testX = testX.reshape((testX.shape[0], 28, 28, 1))

class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', "Sneaker", 'Bag', 'Ankle boot'
]

model = tf.keras.Sequential([ #input_shape 넣어 줘야 summery(모델 아웃라인)보기 가능
    tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)),
    #ndim = 4, Conv2D는 4차원의 데이터를 필요로 한다 [ [ [], [] ... ] ]
    #Convolution layer를 통해 32개의 이미지 복사본 생성, (3, 3)짜리 커널
    #컨볼루젼을 적용하게 되면 이미지가 작아짐 padding을 통해 크기 유지, 활성 함수 relu 이미지를 숫자화 하면 0 ~ 255
    #relu는 음수데이터는 0으로 만든다, 이미지 에서 음수는 있을 수 없기 때문
    tf.keras.layers.MaxPool2D( (2, 2) ),
    #풀링을 통해 사이즈를 더 줄여주고 중요한 정보를 가운데로 모아 준다
    #2, 2크기인 풀링 사이즈

    #tf.keras.layers.Dense(128, input_shape=(28, 28), activation="relu"), #relu: 음수는 다 0으로 만든다
    tf.keras.layers.Flatten(), #2차원 데이터를 1차원으로 압축해주는 레이어
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

'''
결과를 0 ~ 1로 예측하고 싶을때 softmax나  sigmoid를 사용
sigmoid: binary 예측 문제에 사용(0인지? 1인지? / 대학원 붙는다? 안 붙는다?) 마지막 레이어 개수 = 1
softmax: 카테고리 예측 문제에 사용(이옷은 어떤 종류의 옷 일까요?) 마지막 레이어 개수 = 카테고리 수 만큼
'''

model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy']) #카테고리 예측에 쓰는 손실함수
'''
categorical_crossentropy: trainY가 원 핫 인코딩 돼 있을 때
sparse_categorical_crossentropy: trainY가 정수로 돼 있을 때 0, 1, 2 ,3 ...
'''
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=5)
#validation_data: epoch마다 평가해줌
#over fitting을 예방 할 수 있다
#over fitting: 딥러닝 모델이 반복적으로 학습한 데이터의 결과를 외워버리는것 그때부턴 유의미한 학습결과가 나오지 않음
#*결론: val_accuracy를 높일 방법을 찾자

#score = model.evaluate(testX, testY) #학습 후 모델 평가하기
#print(score)