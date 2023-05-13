import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Đọc dữ liệu từ file u.data trong tập dữ liệu MovieLens 100K
data = pd.read_csv('C:/Users/ASUS/Desktop/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

# Đọc dữ liệu về thông tin phim từ file u.item
movies = pd.read_csv('C:/Users/ASUS/Desktop/u.item', sep='|', encoding='latin-1', usecols=[0, 1], names=['item_id', 'title'])

# Tạo bảng dữ liệu chứa thông tin về đánh giá phim của người dùng và thông tin về phim
data = pd.merge(data, movies, on='item_id')
print(data) 

# Tính trung bình xếp hạng cho mỗi phim
ratings = data.groupby('title')['rating'].mean()
ratings['number_of_ratings'] = data.groupby('title')['rating'].count()
print(ratings)

# Tính toán độ tương đồng giữa các bộ phim dựa trên các ratings của người dùng
movie_matrix = data.pivot_table(index='user_id', columns='title', values='rating')

# Giới thiệu phim tương tự như AFO và Contact 
AFO_user_rating = movie_matrix['Air Force One (1997)']
contact_user_rating = movie_matrix['Contact (1997)'] 
print(AFO_user_rating)
print(contact_user_rating)

# Tính mối tương quan giữa xếp hạng của từng bộ phim và của phim AFO 
similar_to_air_force_one=movie_matrix.corrwith(AFO_user_rating)
print(similar_to_air_force_one)

# Tính mối tương quan giữa xếp hạng của từng bộ phim và của phim Contact 
similar_to_contact = movie_matrix.corrwith(contact_user_rating)
print(similar_to_contact)

# Bỏ qua các giá trị null 
corr_contact = pd.DataFrame(similar_to_contact, columns=['Correlation'])
corr_contact.dropna(inplace=True)
print(corr_contact.head)

corr_AFO = pd.DataFrame(similar_to_air_force_one, columns=['Correlation'])
corr_AFO.dropna(inplace=True)
print(corr_AFO)

# Thiết lập ngưỡng cho số lượng xếp hạng > 100 và xem 10 phim đầu tiên 
corr_AFO = corr_AFO.join(ratings['number_of_ratings'])
corr_contact = corr_contact.join(ratings['number_of_ratings'])
print(corr_AFO)
print(corr_contact)

print(corr_AFO[corr_AFO['rating'] > 100].sort_values(by='Correlation', ascending=False)[:10])
print(corr_contact[corr_contact['rating'] > 100].sort_values(by='Correlation', ascending=False).head(10))