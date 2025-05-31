# Laporan Proyek Machine Learning - Hans Kristiandi

## Domain Proyek

### Latar Belakang

Abad 21 menunjukkan perkembangan yang luar biasa, khususnya dalam bidang teknologi. Contoh dari perkembangan teknologi tersebut salah satunya dapat ditemui pada aplikasi streaming film yang dapat diakses melalui HP, tablet, laptop, ataupun TV. Untuk meraih keuntungan yang lebih besar dan memberikan pengalaman terbaik kepada konsumen, banyak dari aplikasi penyedia jasa ini yang menggunakan sebuah fitur yang disebut sistem rekomendasi. Sistem rekomendasi adalah suatu sistem yang merekomendasikan sesuatu terhadap konsumen berdasarkan data perilaku atau preferensi dari waktu ke waktu [1]. Dalam proyek kali ini, akan diteliti lebih lanjut bagaimana sistem rekomendasi bekerja dan cara membuat sistem rekomendasi sederhana untuk merekomendasikan film kepada calon user.

### Mengapa dan bagaimana masalah tersebut harus diselesaikan

Sebuah perusahaan memiliki tujuan untuk mendapatkan keuntungan yang sebesar-besarnya dengan memanfaatkan berbagai peluang yang ada. Konsep tersebut juga berlaku untuk aplikasi penyedia jasa streaming film yang tentunya ingin memberikan pelayanan terbaik kepada konsumen agar tetap setia kepada produk yang ditawarkan. Berbagai riset dilakukan dan salah satunya menunjukkan bahwa konsumen Netflix pada umumnya kehilangan minat setelah memilih selama 60-90 detik dan melihat-lihat 10-20 pilihan film [2]. Untuk mengatasi masalah ini, diperlukan solusi yang konkret dimana salah satunya yaitu dengan menggunakan sistem rekomendasi untuk memberikan saran film yang sesuai dengan preferensi setiap konsumen.

Meski begitu, untuk merancang sebuah sistem rekomendasi bukanlah hal yang mudah. Hal ini dikarenakan banyaknya variabel yang harus dipertimbangkan saat ingin membuat sebuah model, seperti judul film, genre, rating pengguna, aktor, dan sebagainya. Beruntungnya, kemajuan dalam ilmu pengetahuan menghasilkan sebuah penemuan yang dinamakan machine learning. Dengan menggunakan konsep seperti deep learning dan matrix factorization, kita dimampukan untuk merancang sebuah sistem rekomendasi yang mempertimbangkan berbagai variabel dalam proses pelatihannya dan tentunya dapat memberikan hasil akhir yang baik.

### Referensi

[1] F. Kane, "Building Recommender Systems with Machine Learning and AI," _O'Reilly Media_, 2018. Tersedia: https://www.oreilly.com/videos/building-recommender-systems/9781789803273/

[2] C. A. Gomez-Uribe dan N. Hunt, "The Netflix recommender system: Algorithms, business value, and innovation," _ACM Transactions on Management Information System_, vol. 6, no. 4, pp. 1-19, 2015. Tersedia: https://dl.acm.org/doi/abs/10.1145/2843948

## Business Understanding

### Problem Statements

Berdasarkan latar belakang di atas, pernyataan masalah yang dimiliki yaitu:
- Bagaimana membangun model sistem rekomendasi film dengan memanfaatkan deep learning?
- Apakah model yang telah dibangun dapat digunakan untuk merekomendasikan film kepada pengguna?

### Goals

Berdasarkan pernyataan masalah yang telah dibuat, tujuannya yaitu:
- Membuat model untuk memberikan rekomendasi film dengan memanfaatkan deep learning.
- Meodel yang telah dibuat dapat memberikan rekomendasi film secara nyata kepada pengguna.

## Data Understanding

Data yang digunakan pada proyek ini diunduh dari Kaggle, dengan alamat URL yaitu https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system. Dataset ini berasal dari MovieLens, sebuah dataset yang banyak digunakan di bidang sistem rekomendasi, yang berisi tentang peringkat film dan metadata. Dataset ini terdiri atas dua file csv yaitu movies.csv dan ratings.csv dengan total ukuran file sekitar 681 MB. File movies.csv memiliki 62,423 baris dan 3 kolom sementara ratings.csv memiliki 25,000,095 baris dan 4 kolom. Selain itu, didapati juga bahwa data pada dataset sudah bersih karena tidak mengandung nilai null dan nilai duplikat. Adapun movies.csv memiliki kolom sebagai berikut:
- movieId -> No id unik untuk setiap film. Bertipe integer
- title -> Judul film. Bertipe object
- genres -> Genre film (bisa lebih dari 1). Bertipe object

Sementara itu kolom pada ratings.csv yaitu sebagai berikut:
- userId -> No id unik untuk setiap konsumen. Bertipe integer
- movieId -> No id unik untuk setiap film. Bertipe integer
- rating -> Rating film. Bertipe float
- timestamp -> Waktu saat rating diberikan. Bertipe integer

Setelah melalui berbagai tahapan EDA, dihasilkan DataFrame yang merupakan hasil gabungan dari movies dan ratings dengan kolom sebagai berikut:
- movieId
- title
- genres
- userId
- rating
- timestamp

### Variable/feature
Dari enam kolom yang ada pada DataFrame movie_rating, variabel/fitur yang dipilih yaitu:
- userId
- movieId
  
dengan kolom target (label) yaitu **rating**.

### Exploratory Data Analysis

```ruby
movies.head()
ratings.head()
movie_rating.head()
```
Bertujuan untuk mendapatkan gambaran tentang dataset dengan menampilkan lima baris teratas pada masing-masing DataFrame.

```ruby
print('Jumlah data film yang tersedia dalam dataset adalah', len(movies.movieId.unique()))
print('Jumlah data kombinasi genres yang tersedia dalam dataset adalah', len(movies.genres.unique()))
print('Jumlah data user yang memberikan rating dalam dataset adalah', len(ratings.userId.unique()))
print('Jumlah data film yang telah diberikan rating dalam dataset adalah', len(ratings.movieId.unique()))
print('Jumlah pilihan rating yang dapat diberikan adalah', len(ratings.rating.unique()))
```
Bertujuan untuk mengetahui data-data penting seperti jumlah film, jumlah film yang mendapat rating, jumlah pengguna yang memberi rating, jumlah kombinasi genres, dan jumlah pilihan rating yang dapat diberi pengguna.

```ruby
movies.info()
ratings.info()
```
Menampilkan informasi dasar tentang dataset seperti jumlah baris dan tipe data pada masing-masing kolom.

```ruby
movies.isnull().sum()
ratings.isnull().sum()
movie_rating.isna().sum()
```
Melihat apakah dataset memiliki nilai null. Hasilnya tidak didapati nilai null.

```ruby
movies_duplicate_count = movies.duplicated().sum()
ratings_duplicate_count = ratings.duplicated().sum()
duplicate_count = movie_rating.duplicated().sum()
```
Melihat apakah dataset memiliki data duplikat. Hasilnya tidak didapati data duplikat.

```ruby
ratings_list = sorted(float(r) for r in ratings.rating.unique())
```
Menampilkan pilihan rating yang dapat dipilih dan mengurutkannya dari terkecil ke terbesar.

```ruby
rating_counts = ratings['rating'].value_counts().sort_index()
```
Menghitung jumlah user yang memilih rating tersebut

```ruby
ratings['rating'].describe()
```
Melihat distribusi rating seperti nilai median, mean, standar deviasi, Q1, Q3, min, dan max.

### Data Visualization
![download](https://github.com/user-attachments/assets/862ec7c0-0dae-46e1-8b90-d9f7a0fcfa95)

Diagram batang di atas menunjukkan jumlah user yang memberikan rating tertentu. Terlihat bahwa lebih dari 6,000,000 user memberikan rating 4.0 sehingga rating ini adalah yang paling banyak diberi. Sementara itu, terdapat kurang dari 500,000 user yang memberikan rating 0.5 sehingga rating ini adalah jumlah yang paling sedikit diberi.

## Data Preparation

Berbagai tahapan yang dilakukan untuk menyiapkan data agar siap dipakai untuk pelatihan model yaitu:

```ruby
movie_rating = pd.merge(movies, ratings, on='movieId', how='right')
```
Menggabungkan DataFrame movies ke ratings berdasarkan kolom movieId untuk menyiapkan DataFrame yang akan digunakan dalam pelatihan model

```ruby
movie_rating.sort_values('movieId', ascending=True)
```
Menyortir DataFrame berdasarkan kolom movieId dari nilai terkecil ke terbesar agar data menjadi rapi

```ruby
check = movie_rating.groupby('title')[['movieId', 'genres']].nunique()
non_unique = check[(check['movieId'] > 1) | (check['genres'] > 1)]
```
Mengecek apakah sebuah title memiliki tepat satu movieId dan satu genres. Hal ini penting untuk menghindari adanya data duplikat dan miss informasi. Hasilnya didapati ada 89 sampel data yang memiliki lebih dari satu movieId dan genres.

```ruby
fix = (
    movie_rating.groupby('title').agg({
        'movieId': lambda x: x.mode().iloc[0],
        'genres': lambda x: x.mode().iloc[0]
    })
)
```
Memilih nilai modus untuk movieId dan genres dari sebuah title yang sama.

```ruby
movie_rating['movieId'] = movie_rating['title'].map(fix['movieId'])
movie_rating['genres'] = movie_rating['title'].map(fix['genres'])
```
Menyamakan movieId dan genres untuk satu title yang sama dengan menggunakan nilai modus yang telah dicari sebelumnya.

```
if non_unique.empty:
    print("Semua title memiliki tepat satu movieId dan satu genres")
else:
    print("Ada title yang memiliki lebih dari satu movieId atau lebih dari satu genres")
```
Melakukan pengecekan ulang untuk memastikan setiap title memiliki tepat satu movieId dan satu genres.

```ruby
user_ids = df['userId'].unique().tolist()
movie_ids = df['movieId'].unique().tolist()
```
Membuat list untuk masing-masing variabel terpilih dan hanya dapat diisi oleh data unik (tidak ada pengulangan nilai).

```ruby
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
movie_to_movie_encoded = {x: i for i, x in enumerate(movie_ids)}
```
Melakukan encode untuk userId dan movieId untuk mempermudah saat proses training.

```ruby
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
movie_encoded_to_movie = {i: x for i, x in enumerate(movie_ids)}
```
Melakukan inverse encode untuk mengembalikannya ke bentuk semula.

```ruby
df['user'] = df['userId'].map(user_to_user_encoded)
df['movie'] = df['movieId'].map(movie_to_movie_encoded)
```
Memetakan userId dan movieId ke DataFrame yang berkaitan.

```ruby
num_users = len(user_to_user_encoded)
num_movie = len(movie_encoded_to_movie)
min_rating = min(df['rating'])
max_rating = max(df['rating'])
print('Jumlah user: {}, Jumlah film: {}, Min Rating: {}, Max Rating: {}'.format(
    num_users, num_movie, min_rating, max_rating
```
Menampilkan jumlah user yang memberi rating, jumlah film yang mendapat rating, dan rating minimum maupun maksimum yang ada.

```
movie_id = df['movieId'].tolist()
movie_title = df['title'].tolist()
movie_genres = df['genres'].tolist()
```
Menyimpan data pada masing-masing kolom dalam bentuk list.

```ruby
movies = pd.DataFrame({
    'id': movie_id,
    'title': movie_title,
    'genres': movie_genres
```
Membuat key dan menyimpan data ke dalam dictionary dari list yang telah dibuat sebelumnya.

```ruby
movies
```
Menampilkan isi dari dictionary yang telah dibuat.

```ruby
df = df.sample(frac=1, random_state=42)
```
Mengacak data untuk menghindari bias saat splitting data.

```ruby
x = df[['user', 'movie']].values
y = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)
```
Menyimpan data user dan movie ke dalam x (variabel/fitur) dan rating sebagai y (label). Selanjutnya, data dipisah dengan proporsi 80% untuk data latihan dan 20% untuk data validasi.

## Modeling

### Model Training

Pelatihan model menggunakan pendekatan collaborative filtering dengan tahapan sebagai berikut:

```ruby
def __init__(self, num_users, num_movie, embedding_size, **kwargs)
```
Bertujuan untuk menerima jumlah user dan movie sebagai input dan membangun layer embedding.

```ruby
def call(self, inputs):
```
Bertujuan untuk memproses data masuk.

```ruby
model = RecommenderNet(num_users, num_movie, 50)
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
```
Menggunakan arsitektur RecommenderNet dengan embedding size 50, fungsi loss BinaryCrossentropy, optimizer Adam, dan metrik evaluasi berupa RMSE.

```ruby
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
)
```
Membuat fungsi callback untuk menghentikan pelatihan jika val_loss tidak membaik selama 5 epoch berturut-turut.

```ruby
history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 2048,
    epochs = 20,
    validation_data = (x_val, y_val),
    callbacks=[early_stop]
)
```
Melatih model dengan data training, batch size 2048 sampel, maksimal 20 epoch, fungsi callback, dan divalidasi dengan data validasi.

### Result

```ruby
user_id = df.userId.sample(1).iloc[0]
movie_watched_by_user = df[df.userId == user_id]
```
Mengambil satu user dan mencatat history film yang pernah ditonton.

```ruby
movie_not_watched = movies[~movies['id'].isin(movie_watched_by_user.movieId.values)]['id']
```
Mencatat film yang belum pernah ditonton oleh user tersebut. 

```ruby
movie_not_watched = [[movie_to_movie_encoded.get(x)] for x in movie_not_watched]
user_encoder = user_to_user_encoded.get(user_id)
user_movie_array = np.hstack(
    ([[user_encoder]] * len(movie_not_watched), movie_not_watched)
)
```
Membuat array untuk menyimpan film yang belum pernah ditonton.

```ruby
ratings = model.predict(user_movie_array).flatten()
```
Memprediksi rating film (yang belum pernah ditonton) yang akan diberikan oleh user.

```ruby
sorted_indices = ratings.argsort()[::-1]
```
Menyortir 10 film dengan rating tertinggi yang diberikan user.

```ruby
for idx in sorted_indices:
    movie_id = movie_encoded_to_movie.get(movie_not_watched[idx][0])
    if movie_id and movie_id not in seen:
        recommended_movie_ids.append(movie_id)
        seen.add(movie_id)
    if len(recommended_movie_ids) == 10:
        break
```
Menyimpan 10 film dengan rating tertinggi yang mungkin akan diberikan user.

```ruby
top_movie_user = (
    movie_watched_by_user.sort_values(by='rating', ascending=False)
    .head(5)
    .movieId.values
)
```
Menampilkan 5 film dengan rating tertinggi yang pernah diberikan user sebagai data historis.

```ruby
recommended_movie = movies.drop_duplicates(subset='id')
recommended_movie = recommended_movie[recommended_movie['id'].isin(recommended_movie_ids)]
for row in recommended_movie.itertuples():
    print(row.title, ':', row.genres)
```
Menampilkan 10 rekomendasi dan memastikan tidak ada data duplikat dalam rekomendasi tersebut. Data dalam rekomendasi yang ditampilkan yaitu title dan genres.

![Screenshot 2025-05-30 044718](https://github.com/user-attachments/assets/e0c2bb8a-8f4d-47c9-9409-6b19f19ea358)

## Evaluation

![Screenshot 2025-05-30 043837](https://github.com/user-attachments/assets/7767d464-c655-4299-a865-c6209ffaf37a)

Setelah 6 epoch, pelatihan dihentikan dan diperoleh training loss 4.48, RMSE training 0.48, validation loss 5.26, dan RMSE validasi 0.42.

![download (1)](https://github.com/user-attachments/assets/44c79afa-1eda-4e29-abdb-c03d81ce9d16)

Jika dilihat, RMSE training > RMSE validasi. Hal ini kemungkinan disebabkan oleh batch size yang terlalu besar dan pelatihan epoch yang terlalu singkat. Meski begitu, RMSE validasi yang diperoleh yaitu 0.42 terbilang cukup baik untuk dataset dengan jumlah data lebih dari 25,000,000.

### Evaluation Metric

Metrik yang digunakan untuk mengevaluasi pelatihan model yaitu Root Mean Squared Error (RMSE). Metrik ini digunakan untuk mengukur seberapa jauh prediksi model dari nilai sebenarnya. Pada dasarnya RMSE dapat dihitung sebagai akar dari Mean Squared Error (MSE), sehingga metrik ini memberikan bobot lebih pada error yang besar karena rumus yang dimilikinya. Semakin kecil nilai RMSE, maka model memiliki akurasi yang lebih tinggi.

![download (1)](https://github.com/user-attachments/assets/d5de7b21-e927-4a01-b782-f3e2ebeda287)


**---Ini adalah bagian akhir laporan---**



