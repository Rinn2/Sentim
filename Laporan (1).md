# Laporan Proyek Machine Learning - Agus Irvan Maulana

## Project Overview
Proyek ini membuat sistem rekomendasi buku yang membantu pengguna menemukan buku yang cocok dengan minat dan kebutuhan mereka. Saat ini, banyak sekali pilihan buku digital, sehingga pengguna sering kesulitan mencari buku yang tepat. Hal ini membuat pengalaman membaca menjadi kurang menyenangkan dan kurang sesuai dengan keinginan.

Masalah ini penting karena jumlah buku digital terus bertambah, dan pengguna ingin mendapat rekomendasi yang benar-benar relevan. Untuk itu, proyek ini menggunakan dua cara utama dalam memberikan rekomendasi:

- Content-Based Filtering untuk merekomendasikan buku dengan karakteristik serupa dengan yang disukai pengguna.
- Collaborative Filtering untuk menyarankan buku berdasarkan perilaku pengguna lain yang mirip.



## Business Understanding
### Problem Statements
Pengguna sering kali kesulitan menemukan buku yang sesuai dengan minat dan kebutuhannya karena banyaknya pilihan yang tersedia. Hal ini menyebabkan :
- Rendahnya keterlibatan pengguna
- Menurunnya kepuasan pengguna dalam menggunakan platform
- Kurangnya personalisasi dalam pengalaman membaca
  
### Goals
- Memberikan rekomendasi buku yang relevan dan dipersonalisasi untuk setiap pengguna
- Meningkatkan user engagement dan waktu yang dihabiskan dalam platform
- Meningkatkan konversi terhadap pembelian atau pembacaan buku



### Solution statements
#### Content-Based Filtering
- Menggunakan metadata buku seperti judul, penulis, kategori, bahasa, dan genre
- Menganalisis kesamaan antara buku berdasarkan deskripsi kontennya (TF-IDF, cosine similarity, dll.)
- Memberikan rekomendasi buku yang mirip dengan yang sudah disukai oleh pengguna

#### Collaborative Filtering
- Menggunakan data interaksi pengguna (user_id, isbn, dan rating)
- Membangun model dengan pendekatan matrix factorization 
- Merekomendasikan buku berdasarkan preferensi pengguna lain yang mirip
  
## Data Understanding
Dataset yang digunakan berisi 1.031.175 entri yang mencatat interaksi pengguna terhadap buku.data ini berasal dari [kaggle](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset?select=Books+Data+with+Category+Language+and+Summary) .

Variabel-variabel pada dataset adalah sebagai berikut:
- Unnamed: 0 : Index bawaan dari proses penyimpanan/penggabungan data
- user_id : ID unik pengguna. Digunakan untuk mengidentifikasi siapa yang membaca atau memberi rating pada buku.
- location : Informasi lokasi pengguna dalam format gabungan (biasanya City, State, Country)
- age : usia pengguna
- isbn : Nomor identifikasi unik untuk setiap buku (International Standard Book Number)
- rating : Nilai rating yang diberikan oleh pengguna terhadap buku
- book_title : Judul  dari buku
- book_author : Nama penulis buku
- year_of_publication : Tahun terbit buku
- Nama penerbit buku
- img_s/img_m/img_l : URL gambar sampul buku ukuran kecil, sedang, besar
- Summary : Ringkasan/deskripsi buku
- Language : Bahasa buku
- Category : Kategori/genre buku
- city : Nama kota penggun
- state : Nama negara bagian pengguna
- country : Negara pengguna

#### Jumlah missing value pada dataset :
![image](https://github.com/user-attachments/assets/aa7e3e58-d3fc-4390-8078-8649780aa935)

Dataset memiliki missing value pada kolom book_author, city, state, dan country, dengan jumlah terbanyak pada country yaitu 35.374 data yang hilang.

#### Jumlah Duplikat Data
![image](https://github.com/user-attachments/assets/ceab0796-daef-4ed2-aa97-d5ec993a693c)

Dataset tidak memiliki data duplikat, sehingga seluruh baris bersifat unik.

## Data Preparation
### Data Preparation Content Based Filtering
#### 1.menghapus missing value
Menghapus semua baris yang memiliki nilai kosong (missing value) agar tidak mengganggu proses modeling, terutama dalam kolom penting seperti book_author, city, state, dan country

#### 2. Melakukan sampling data 
menggunakan 20000 data yang ada dan Language bahasa inggris yang digunakan untuk membuat model

#### 3. Konversi Kolom menjadi list  dan membuat dictionary data
Langkah ini mengonversi kolom isbn, book_title, dan category menjadi list, lalu membentuk dictionary data berisi 20.000 buku berbahasa Inggris sebagai dasar pemodelan.

#### 4.TF-IDF Vectorization
Mengubah kategori buku menjadi representasi vektor menggunakan TF-IDF, lalu menghitung cosine similarity antar buku untuk Content-Based Filtering

### Data Preparation Collaborative Filtering
#### 1. Drop buku dengan rating 0
Baris dengan rating = 0 dihapus karena dianggap tidak memberikan evaluasi terhadap buku
#### 2.Melakukan Encoding
Encoding dilakukan dengan mengubah nilai kategorikal user_id dan isbn menjadi angka agar dapat digunakan oleh model. Setiap user_id dan isbn dipetakan ke angka unik menggunakan dictionary (encode_user_id dan encode_book_id), serta disediakan mapping sebaliknya (decode_user_id dan decode_book_id) untuk interpretasi hasil prediksi. Hasil encoding ini disimpan ke dalam kolom baru user dan book pada 
#### 3. Normaliasi Rating
Melakukan normalisasi nilai rating ke rentang 0–1 agar sesuai dengan fungsi aktivasi sigmoid dalam model Collaborative Filtering
DataFrame.
#### 4.Splitting Dataset
Data dibagi menjadi 80% data latih dan 20% data validasi untuk mengevaluasi kinerja model.

## Modeling
### Content Based Filtering
Content-Based Filtering adalah salah satu teknik dalam sistem rekomendasi yang menggunakan informasi karakteristik konten dari setiap item untuk memberikan rekomendasi. Sistem ini menganalisis fitur-fitur dari item yang sudah disukai oleh pengguna, lalu mencari item lain yang memiliki fitur serupa untuk direkomendasikan.
### Cara Kerja model 
Dalam sistem ini, representasi konten item (buku) diekstrak menggunakan TF-IDF Vectorizer pada fitur category. Setiap buku direpresentasikan sebagai vektor dalam ruang berdimensi tinggi. Kemudian, cosine similarity digunakan untuk mengukur tingkat kemiripan antara dua buku berdasarkan sudut antara vektor fitur mereka. Semakin kecil sudutnya, semakin besar kemiripannya. Buku-buku dengan nilai cosine similarity tertinggi terhadap buku yang disukai pengguna akan direkomendasikan.
#### Kelebihan 
- Rekomendasi disesuaikan dengan preferensi unik pengguna berdasarkan item yang sudah mereka sukai sebelumnya
- Sistem bisa bekerja tanpa memerlukan data dari pengguna lain
- Bisa merekomendasikan item baru selama fitur item tersebut tersedia

#### Kekurangan  
- Cenderung merekomendasikan item yang sangat mirip dengan yang sudah diketahui pengguna
- Membutuhkan representasi fitur yang lengkap dan berkualitas tinggi agar rekomendasi akurat
- Bisa merekomendasikan item baru selama fitur item tersebut tersedia
- Kadang-kadang sulit menangkap preferensi pengguna yang kompleks

#### Hasil Top-10 Recommendation

![image](https://github.com/user-attachments/assets/f4829fcd-4e6a-444a-bd16-2d58b86831ed)

Hasil yang diprediksi oleh model
### Colaborative Filtering
Collaborative Filtering adalah teknik dalam sistem rekomendasi yang memberikan rekomendasi berdasarkan preferensi atau perilaku pengguna lain yang memiliki kesamaan dengan pengguna target. Sistem ini tidak bergantung pada konten item, melainkan pada pola interaksi pengguna dengan item 
### Cara Kerja 
Model dikembangkan menggunakan TensorFlow Keras dan mengimplementasikan Neural Collaborative Filtering dengan beberapa komponen  berikut:
- Embedding Layer
Mewakili user_id dan book_id dalam bentuk vektor berdimensi rendah (dense vector)
- Dot Product dan bias 
Model menghitung skor interaksi antara user dan book melalui operasi dot product antara dua embedding
- Dropout Layer
Digunakan dengan nilai dropout 0.3 untuk mencegah overfitting dengan mengabaikan sebagian output selama proses pelatihan (training)
- Sigmoid Activation
Output akhir dari model distandarkan ke dalam rentang 0–1 dengan fungsi sigmoid, merepresentasikan tingkat prediksi minat pengguna terhadap buku tertentu.


#### Kelebihan 
- Dapat memberikan rekomendasi yang beragam dan tidak terbatas hanya pada fitur item tertentu
- Mampu menangkap preferensi pengguna yang kompleks melalui pola interaksi komunitas pengguna

#### Kekurangan  
- Membutuhkan data pengguna dan interaksi yang cukup banyak agar rekomendasi bisa akurat 
- Rentan terhadap masalah sparsity, yaitu data interaksi yang sangat sedikit sehingga sulit menemukan kemiripan antar pengguna atau item
- Bisa terjadi bias popularitas, dimana item yang populer lebih sering direkomendasikan sehingga item niche kurang mendapat perhatian

#### Hasil Top-10 Recommendation

![image](https://github.com/user-attachments/assets/2ca3f2af-e2c1-41e0-be31-3f5ef888f794)

Hasil yang diprediksi oleh model

## Evaluation
### Content Based Filtering
Precision adalah salah satu metrik evaluasi yang sering digunakan untuk menilai kualitas sistem rekomendasi, termasuk Content-Based Filtering. Precision mengukur seberapa tepat rekomendasi yang diberikan oleh sistem.

Precision = (Jumlah rekomendasi buku yang relevan) / (Jumlah item yang direkomendasikan)

dalam hasil yang diberikan menggunakan sebagai berikut 
	Precision = $\\frac{10}{10} = 1$

Model berhasil merekomendasikan buku yang mirip berdasarkan kontennya (kategori), sehingga cocok digunakan untuk pengguna baru yang belum banyak berinteraksi dengan sistem. Ini menjawab kebutuhan sistem untuk tetap memberikan rekomendasi bahkan ketika interaksi pengguna masih minim (cold start user).

    
### Colaborative Filtering
Root Mean Square Error (RMSE) digunakan untuk mengukur seberapa dekat prediksi sistem rekomendasi dengan rating sebenarnya yang diberikan oleh pengguna. RMSE menghitung rata-rata kuadrat dari selisih antara rating yang diprediksi dan rating aktual, lalu diakarkan.

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} ( \hat{y}_i - y_i )^2}
$$

Penjelasan :
- $\hat{y}_i$ = nilai prediksi ke-*i*  
- $y_i$ = nilai aktual ke-*i*  
- $n$ = jumlah data

![image](https://github.com/user-attachments/assets/088bf5b6-9de8-408d-a625-207ee74b86b0)


Model collaborative filtering yang dikembangkan berhasil mencapai performa yang cukup baik dengan nilai Root Mean Squared Error (RMSE) sebesar 0.2721 pada data pelatihan dan 0.2683 pada data pengujian. Perbedaan nilai RMSE yang kecil antara data pelatihan dan pengujian menunjukkan bahwa model memiliki kemampuan generalisasi yang baik dan tidak mengalami overfitting secara signifikan.

### Hubungkan dengan Business Understanding
Sistem rekomendasi ini berhasil menjawab permasalahan utama dengan memberikan rekomendasi buku yang sesuai preferensi pengguna. Content-Based Filtering menganalisis kategori buku yang disukai, sedangkan Collaborative Filtering memanfaatkan pola rating pengguna lain yang serupa.

Dari sisi bisnis, sistem ini meningkatkan pengalaman pengguna dan mempermudah pencarian buku. Content-Based cocok untuk pengguna baru, sementara Collaborative Filtering efektif untuk pengguna aktif, sehingga keduanya saling melengkapi dalam mendukung tujuan bisnis.
