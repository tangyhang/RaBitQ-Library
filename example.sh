# compiling 
mkdir build bin 
cd build 
cmake ..
make 

# Download the dataset
wget -P /home/ANNS_SSD/RabitQ/gist ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
tar -xzvf /home/ANNS_SSD/RabitQ/gist/gist.tar.gz -C /home/ANNS_SSD/RabitQ/gist

# indexing and querying for symqg
./bin/symqg_indexing /home/ANNS_SSD/RabitQ/gist/gist_base.fvecs 32 400 /home/ANNS_SSD/RabitQ/gist/symqg_32.index

./bin/symqg_querying /home/ANNS_SSD/RabitQ/gist/symqg_32.index /home/ANNS_SSD/RabitQ/gist/gist_query.fvecs /home/ANNS_SSD/RabitQ/gist/gist_groundtruth.ivecs

# indexing and querying for RabitQ+ with ivf, please refer to python/ivf.py for more information about clustering
python ./python/ivf.py /home/ANNS_SSD/RabitQ/gist/gist_base.fvecs 4096 /home/ANNS_SSD/RabitQ/gist/gist_centroids_4096.fvecs /home/ANNS_SSD/RabitQ/gist/gist_clusterids_4096.ivecs

./bin/ivf_rabitq_indexing /home/ANNS_SSD/RabitQ/gist/gist_base.fvecs /home/ANNS_SSD/RabitQ/gist/gist_centroids_4096.fvecs /home/ANNS_SSD/RabitQ/gist/gist_clusterids_4096.ivecs 3 /home/ANNS_SSD/RabitQ/gist/ivf_4096_3.index

./bin/ivf_rabitq_querying /home/ANNS_SSD/RabitQ/gist/ivf_4096_3.index /home/ANNS_SSD/RabitQ/gist/gist_query.fvecs /home/ANNS_SSD/RabitQ/gist/gist_groundtruth.ivecs

# indexing and querying for RabitQ+ with hnsw, do clustering first
python /home/ANNS_SSD/RabitQ/python/ivf.py /home/ANNS_SSD/RabitQ/gist/gist_base.fvecs 16 /home/ANNS_SSD/RabitQ/gist/gist_centroids_16.fvecs /home/ANNS_SSD/RabitQ/gist/gist_clusterids_16.ivecs

./bin/hnsw_rabitq_indexing /home/ANNS_SSD/RabitQ/gist/gist_base.fvecs /home/ANNS_SSD/RabitQ/gist/gist_centroids_16.fvecs /home/ANNS_SSD/RabitQ/gist/gist_clusterids_16.ivecs 16 200 5 /home/ANNS_SSD/RabitQ/gist/hnsw_5.index

./bin/hnsw_rabitq_querying /home/ANNS_SSD/RabitQ/gist/hnsw_5.index /home/ANNS_SSD/RabitQ/gist/gist_query.fvecs /home/ANNS_SSD/RabitQ/gist/gist_groundtruth.ivecs
