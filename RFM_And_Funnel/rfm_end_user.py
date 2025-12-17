import pandas as pd
import mysql.connector
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
import time
import logging
import os
from sqlalchemy import text

# Setup logging
log_directory = '/root/rfm/logs'
os.makedirs(log_directory, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_directory, 'errors.log'),
    format='%(message)s - %(asctime)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

# Set opsi tampilan pandas
pd.set_option('display.float_format', '{:.0f}'.format)

# Fungsi untuk menghubungkan ke database MySQL
def connect_to_database(host, user, password, database, port):
    return mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        port=port
    )

# Fungsi untuk menarik data dari database
def fetch_data(cursor):
    query = """
    SELECT 
        sales.customer_id AS customer_id,
        sales.last_shipment_check AS last_shipment_date,
        sale_items.sale_date AS invoice_date,
        sale_items.total_price AS product_sales_total,
        sale_items.product_name,
        sale_items.product_id,
        sale_items.product_quantity AS product_qty,
        customers.province_id,
        sales.sale_status AS sale_status
    FROM sales
    LEFT JOIN customers ON sales.customer_id = customers.id
    LEFT JOIN provinces ON customers.province_id = provinces.id
    LEFT JOIN sale_items ON sales.id = sale_items.sale_id
    WHERE sales.channel_id IN (1, 2, 3, 999)
    AND customers.type = 1
    AND sale_items.product_name IN ('BIO INSULEAF 150 ML', 'BIO INSULEAF', 'ETAWALIN', 'ETAWAKU', 'NUTRIFLAKES', 'ZYMUNO 130 ML', 'ZYMUNO')
    """
    cursor.execute(query)
    result = cursor.fetchall()
    return pd.DataFrame(result, columns=['customer_id', 'last_shipment_date', 'invoice_date', 
                                         'product_sales_total', 'product_name', 
                                         'product_id', 'product_qty', 'province_id', 'sale_status'])
def exclude_customer_return(df):
    purchase_frequency = df.groupby(['customer_id', 'product_name']).size().reset_index(name='frequency')
    df_with_frequency = df.merge(purchase_frequency, on=['customer_id', 'product_name'], how='left')
    
    df_high_frequency = df_with_frequency[df_with_frequency['frequency'] > 3].copy()
    return_counts = df_high_frequency[df_high_frequency['sale_status'] == 3].groupby(['customer_id', 'product_name']).size().reset_index(name='return_count')
    df_with_returns = df_high_frequency.merge(return_counts, on=['customer_id', 'product_name'], how='left').fillna(0)
    
    df_with_returns['return_percentage'] = df_with_returns['return_count'] / df_with_returns['frequency'] * 100
    
    return_summary = df_with_returns[['customer_id', 'product_name', 'return_percentage']].drop_duplicates()
    high_return_customers = return_summary[return_summary['return_percentage'] > 50]
    df_with_return_percentage = df.merge(return_summary, on=['customer_id', 'product_name'], how='left')
    df_with_return_percentage['return_percentage'] = df_with_return_percentage['return_percentage'].fillna(0)
    
    final_df = df_with_return_percentage[df_with_return_percentage['return_percentage'] < 50]
    final_df_clean = final_df.drop(columns=['sale_status', 'return_percentage'])
    return final_df_clean

# Fungsi untuk menghitung metrik RFM
def calculate_rfm(df):
    today = datetime.today()
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')
    df['last_shipment_date'] = df['last_shipment_date'].fillna(df['invoice_date'] + pd.Timedelta(days=5))
    df['last_shipment_date'] = pd.to_datetime(df['last_shipment_date'])

    df['given_product_name'] = df['product_name']
    df.loc[df['given_product_name'].str.contains('ZYMUNO', case=False, na=False), 'given_product_name'] = 'ZYMUNO'
    df.loc[df['given_product_name'].str.contains('BIO INSULEAF', case=False, na=False), 'given_product_name'] = 'BIO INSULEAF'

    recency = df.groupby(['customer_id', 'given_product_name'])['last_shipment_date'].max().reset_index()
    recency['recency'] = (today - recency['last_shipment_date']).dt.days
    
    frequency = df.groupby(['customer_id', 'given_product_name']).size().reset_index(name='frequency')
    
    monetary = df.groupby(['customer_id', 'given_product_name'])['product_sales_total'].sum().reset_index(name='monetary')

    last_qty = df.loc[df.groupby(['customer_id', 'given_product_name'])['last_shipment_date'].idxmax()][['customer_id', 'given_product_name', 'product_qty']]
    last_qty = last_qty.groupby(['customer_id', 'given_product_name'])['product_qty'].sum().reset_index()

    rfm = recency.merge(frequency, on=['customer_id', 'given_product_name'])
    rfm = rfm.merge(monetary, on=['customer_id', 'given_product_name'])
    rfm = rfm.merge(last_qty, on=['customer_id', 'given_product_name'])
    
    rfm['recency'] = rfm['recency'].apply(lambda x: max(x, 0))
    
    return rfm

# Fungsi untuk melakukan clustering dan mengklasifikasikan cluster
def cluster_and_classify(rfm, product_name, n_clusters):
    rfm_product = rfm[rfm['given_product_name'] == product_name].reset_index(drop=True)
    
    X = rfm_product[['recency', 'frequency', 'monetary']].reset_index(drop=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    
    rfm_product['cluster'] = kmeans.labels_
    
    cluster_summary = rfm_product.groupby('cluster').agg({
        'frequency': ['median']
    }).reset_index()
    
    cluster_summary.columns = ['cluster', 'frequency_median']
    
    def classify_cluster(median):
        if median >= 6:
            return 'Champion'
        elif 2 <= median < 6:
            return 'Loyal'
        elif median < 2:
            return 'Potential Loyal'
        else:
            return 'Unknown'
    
    cluster_summary['cluster_name'] = cluster_summary['frequency_median'].apply(classify_cluster)
    cluster_mapping = cluster_summary.set_index('cluster')['cluster_name'].to_dict()
    
    rfm_product['cluster'] = rfm_product['cluster'].replace(cluster_mapping)
    rfm_product.loc[(rfm_product['frequency'] > 5), 'cluster'] = 'Champion'

    return rfm_product

def calculate_funnel(df, product_name):
    if product_name == 'BIO INSULEAF':
        df_bio = df[df['given_product_name'] == 'BIO INSULEAF']
        result_bio = df_bio[df_bio.groupby(['customer_id', 'given_product_name'])['last_shipment_date']
                        .transform('max') == df_bio['last_shipment_date']]
        # Pilih kolom yang diinginkan
        result_bio = result_bio[['customer_id', 'last_shipment_date', 'product_name', 'product_qty']].drop_duplicates()
        result_bio['product_qty_bio'] = result_bio['product_qty'].where(result_bio['product_name'] == 'BIO INSULEAF')
        result_bio['product_name_bio'] = result_bio['product_name'].where(result_bio['product_name'] == 'BIO INSULEAF')
        
        result_bio['product_qty_bio_150ml'] = result_bio['product_qty'].where(result_bio['product_name'] == 'BIO INSULEAF 150 ML')
        result_bio['product_name_bio_150ml'] = result_bio['product_name'].where(result_bio['product_name'] == 'BIO INSULEAF 150 ML')
        
        # Grupkan berdasarkan 'customer_id' dan 'last_shipment_date'
        result_bio = result_bio.groupby(['customer_id', 'last_shipment_date']).agg({
            'product_qty_bio': 'max',  # Agregasi numerik
            'product_name_bio': 'first',  # Agregasi teks, bisa juga menggunakan 'max' untuk mendapatkan nilai teks
            'product_qty_bio_150ml': 'max',
            'product_name_bio_150ml': 'first'
        }).reset_index()
        
        lama_habis = 21
        lama_habis_bio150ml = 12
        # today = datetime.strptime(end_date, '%Y-%m-%d')
        today = datetime.today()
        three_months_ago = today - pd.DateOffset(months=3)
        
        # Fungsi untuk menentukan funnel dan est_habis_produk
        def determine_funnel(row):
            shipment_plus_qty = timedelta(days=0)  # Inisialisasi dengan timedelta nol
        
            # Cek jika produk bio ada
            if pd.notna(row['product_name_bio']):
                shipment_plus_qty += timedelta(days=row['product_qty_bio'] * lama_habis)
        
            # Cek jika produk bio150ml ada
            if pd.notna(row['product_name_bio_150ml']):
                shipment_plus_qty += timedelta(days=row['product_qty_bio_150ml'] * lama_habis_bio150ml)
        
            # Hitung tanggal pengiriman plus kuantitas
            final_shipment_date = row['last_shipment_date'] + shipment_plus_qty
        
            # Tentukan funnel berdasarkan final_shipment_date
            if final_shipment_date >= today:
                funnel_status = 'Hot'
            elif final_shipment_date < today and final_shipment_date >= three_months_ago:
                funnel_status = 'Warm'
            else:
                funnel_status = 'Cold'
            
            # Kembalikan funnel dan final_shipment_date
            return pd.Series([funnel_status])
        
        # Terapkan fungsi ke DataFrame
        result_bio[['funnel']] = result_bio.apply(determine_funnel, axis=1)
        return result_bio
    elif product_name == 'ZYMUNO':
        df_zymuno = df[df['given_product_name'] == 'ZYMUNO']
        result_zymuno = df_zymuno[df_zymuno.groupby(['customer_id', 'given_product_name'])['last_shipment_date']
                        .transform('max') == df_zymuno['last_shipment_date']]
        # Pilih kolom yang diinginkan
        result_zymuno = result_zymuno[['customer_id', 'last_shipment_date', 'product_name', 'product_qty']].drop_duplicates()
        result_zymuno['product_qty_zymuno'] = result_zymuno['product_qty'].where(result_zymuno['product_name'] == 'ZYMUNO')
        result_zymuno['product_name_zymuno'] = result_zymuno['product_name'].where(result_zymuno['product_name'] == 'ZYMUNO')
        
        result_zymuno['product_qty_zymuno_130ml'] = result_zymuno['product_qty'].where(result_zymuno['product_name'] == 'ZYMUNO 130 ML')
        result_zymuno['product_name_zymuno_130ml'] = result_zymuno['product_name'].where(result_zymuno['product_name'] == 'ZYMUNO 130 ML')
        
        # Grupkan berdasarkan 'customer_id' dan 'last_shipment_date'
        result_zymuno = result_zymuno.groupby(['customer_id', 'last_shipment_date']).agg({
            'product_qty_zymuno': 'max',  # Agregasi numerik
            'product_name_zymuno': 'first',  # Agregasi teks, bisa juga menggunakan 'max' untuk mendapatkan nilai teks
            'product_qty_zymuno_130ml': 'max',
            'product_name_zymuno_130ml': 'first'
        }).reset_index()
        
        lama_habis = 10
        lama_habis_zymuno130ml = 6
        # today = datetime.strptime(end_date, '%Y-%m-%d')
        today = datetime.today()
        three_months_ago = today - pd.DateOffset(months=3)
        
        # Fungsi untuk menentukan funnel dan est_habis_produk
        def determine_funnel(row):
            shipment_plus_qty = timedelta(days=0)  # Inisialisasi dengan timedelta nol
        
            # Cek jika produk zymuno ada
            if pd.notna(row['product_name_zymuno']):
                shipment_plus_qty += timedelta(days=row['product_qty_zymuno'] * lama_habis)
        
            if pd.notna(row['product_name_zymuno_130ml']):
                shipment_plus_qty += timedelta(days=row['product_qty_zymuno_130ml'] * lama_habis_zymuno130ml)
        
            # Hitung tanggal pengiriman plus kuantitas
            final_shipment_date = row['last_shipment_date'] + shipment_plus_qty
        
            # Tentukan funnel berdasarkan final_shipment_date
            if final_shipment_date >= today:
                funnel_status = 'Hot'
            elif final_shipment_date < today and final_shipment_date >= three_months_ago:
                funnel_status = 'Warm'
            else:
                funnel_status = 'Cold'
            
            # Kembalikan funnel dan final_shipment_date
            return pd.Series([funnel_status])
        
        # Terapkan fungsi ke DataFrame
        result_zymuno[['funnel']] = result_zymuno.apply(determine_funnel, axis=1)
        return result_zymuno
    else:
        df = df[df['given_product_name'] == product_name]
        consumption_days = {
            'ETAWALIN': 7,
            'NUTRIFLAKES': 5,
            'ETAWAKU': 8
        }
        
        # Set the consumption days for the given product
        lama_habis = consumption_days.get(product_name)
        
        # Convert last_shipment_date to datetime
        df['last_shipment_date'] = pd.to_datetime(df['last_shipment_date'])
        
        # Set today’s date and three-months-ago threshold
        today = datetime.today()
        three_months_ago = today - pd.DateOffset(months=3)

        # Function to determine funnel status and estimated depletion date
        def determine_funnel(row):
            est_depletion_date = row['last_shipment_date'] + timedelta(days=row['product_qty'] * lama_habis)

            # Determine funnel status
            if est_depletion_date >= today:
                funnel_status = 'Hot'
            elif est_depletion_date >= three_months_ago:
                funnel_status = 'Warm'
            else:
                funnel_status = 'Cold'
            
            return pd.Series([funnel_status])

        # Apply the function to each row
        df[['funnel']] = df.apply(determine_funnel, axis=1)
        return df

def repeat_order(df):
    # Fungsi untuk menentukan nilai repeat_order
    def get_repeat_order(frequency):
        if frequency == 1:
            return 0
        elif frequency == 2:
            return 1
        elif frequency == 3:
            return 2
        elif frequency == 4:
            return 3
        elif frequency == 5:
            return 4
        else:
            return 5
    
    # Membuat kolom repeat_order
    df['repeat_order'] = df['frequency'].apply(get_repeat_order)
    return df

# Fungsi untuk menggabungkan semua data produk dan mengupload ke MySQL
def upload_to_mysql(engine, result):
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE rfm"))
        
    result.to_sql(name='rfm', con=engine, if_exists='append', index=False, chunksize=1000)

def posprocessing_cluster(df):
    # df.loc[(df['frequency'] > 5), 'cluster'] = 'Champion'
    df.loc[(df['cluster'] == 'Potential Loyal') & (df['funnel'] == 'Cold'), 'cluster'] = 'Hibernate'

    return df

def main():
    try:
        start_time = time.time()
        mydb = connect_to_database(
          host="arjuna.genah.net",
          user="arjuna_read_only",
          password="AcgK_GWB-24",
          database="arjuna",
          port = 33061
        )
        mycursor = mydb.cursor()
        df = fetch_data(mycursor)
        print("Data retrieval completed!")
        df = exclude_customer_return(df)
        
        rfm = calculate_rfm(df)
        
        rfm_bio = cluster_and_classify(rfm, 'BIO INSULEAF', n_clusters=5)
        rfm_etawalin = cluster_and_classify(rfm, 'ETAWALIN', n_clusters=5)
        rfm_etawaku = cluster_and_classify(rfm, 'ETAWAKU', n_clusters=5)
        rfm_nutriflakes = cluster_and_classify(rfm, 'NUTRIFLAKES', n_clusters=5)
        rfm_zymuno = cluster_and_classify(rfm, 'ZYMUNO', n_clusters=5) 
        
        funnel_bio = calculate_funnel(df, 'BIO INSULEAF')[['customer_id', 'last_shipment_date', 'funnel']]
        funnel_etawalin = calculate_funnel(rfm, 'ETAWALIN')[['customer_id', 'last_shipment_date', 'funnel']].drop_duplicates(subset = 'customer_id')
        funnel_etawaku = calculate_funnel(rfm, 'ETAWAKU')[['customer_id', 'last_shipment_date', 'funnel']].drop_duplicates(subset = 'customer_id')
        funnel_nutriflakes = calculate_funnel(rfm, 'NUTRIFLAKES')[['customer_id', 'last_shipment_date', 'funnel']].drop_duplicates(subset = 'customer_id')
        funnel_zymuno = calculate_funnel(df, 'ZYMUNO')[['customer_id', 'last_shipment_date', 'funnel']]
        
        join_rfm_bio  = rfm_bio.merge(funnel_bio, on = ['customer_id', 'last_shipment_date'])
        join_rfm_etawalin = rfm_etawalin.merge(funnel_etawalin, on = ['customer_id', 'last_shipment_date'])
        join_rfm_etawaku = rfm_etawaku.merge(funnel_etawaku, on = ['customer_id', 'last_shipment_date'])
        join_rfm_nutriflakes = rfm_nutriflakes.merge(funnel_nutriflakes, on = ['customer_id', 'last_shipment_date'])
        join_rfm_zymuno = rfm_zymuno.merge(funnel_zymuno, on = ['customer_id', 'last_shipment_date'])
        
        df_all = pd.concat([join_rfm_bio, join_rfm_etawalin, join_rfm_etawaku, join_rfm_nutriflakes, join_rfm_zymuno])
        df_all = repeat_order(df_all).rename(columns = {'given_product_name' : 'product_name'})
        
        df_province = df[['customer_id', 'province_id']].drop_duplicates()
        df_product = df[['product_name', 'product_id']].drop_duplicates()
        
        df_result = df_all.merge(df_province, on = ['customer_id'], how = 'left')
        df_result = df_result.merge(df_product, on = ['product_name'], how = 'left')
        df_result = posprocessing_cluster(df_result)
        
        df_result = df_result[['customer_id', 'cluster', 'funnel', 'product_name', 'product_id', 'last_shipment_date',
              'product_qty', 'repeat_order', 'province_id']]  
        
        engine = create_engine('mysql+mysqlconnector://remote:COE_Remote%4024%21@coegenah.com:33061/rfm_automation')
        upload_to_mysql(engine, df_result)
        
        end_time = time.time()
        execution_time = end_time - start_time
        hours, rem = divmod(execution_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Execution time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
        logging.info("ETL code complete. All data have been store successfully.")
    except TypeError as te:
        logging.warning(f"TypeError: {te}")
    except mysql.connector.Error as db_err:
        logging.error(f"Database error: {db_err}", exc_info=True)
    except pd.errors.MergeError as merge_err:
        logging.error(f"Data merge error: {merge_err}", exc_info=True)
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        raise


# Jalankan fungsi utama
if __name__ == "__main__":
    main()