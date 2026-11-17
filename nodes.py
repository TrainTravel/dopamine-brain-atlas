# Approximate MNI152 centroids (right hemisphere; mirrored for left). mid=True -> midline structure.
# expr: qualitative receptor density 0-3 for D1..D5 from the receptor-mapping literature (autoradiography, PET, mRNA).
NODES = [
 # id, name, (x,y,z), mid, role, expr[D1..D5]
 ("VTA","Ventral tegmental area",(4,-18,-14),False,"source",[0,2,2,0,1]),
 ("SNc","Substantia nigra",(10,-18,-12),False,"source",[1,3,2,0,2]),
 ("HYP","Hypothalamus (arcuate)",(4,-4,-12),False,"source",[1,1,2,2,2]),
 ("PIT","Pituitary",(0,2,-34),True,"target",[0,3,0,0,0]),
 ("NAc","Nucleus accumbens",(10,10,-8),False,"target",[3,3,3,0,1]),
 ("CAU","Caudate",(13,12,10),False,"target",[3,3,1,0,1]),
 ("PUT","Putamen",(26,4,2),False,"target",[3,3,1,0,1]),
 ("GP","Globus pallidus",(18,-2,-2),False,"relay",[1,1,3,0,0]),
 ("STN","Subthalamic nucleus",(12,-14,-6),False,"relay",[1,1,1,0,1]),
 ("THA","Thalamus",(8,-18,8),False,"relay",[0,1,2,1,2]),
 ("AMY","Amygdala",(24,-4,-18),False,"target",[2,1,1,2,1]),
 ("HIP","Hippocampus",(28,-22,-14),False,"target",[1,1,0,2,3]),
 ("ACC","Anterior cingulate / medial PFC",(6,36,18),False,"target",[2,1,0,2,2]),
 ("DLPFC","Dorsolateral prefrontal cortex",(42,32,28),False,"target",[2,1,0,2,2]),
 ("OFC","Orbitofrontal cortex",(24,34,-14),False,"target",[2,1,0,2,1]),
]
PATHS = [
 ("mesolimbic","Mesolimbic","Reward, motivation, salience",[("VTA","NAc"),("VTA","AMY"),("VTA","HIP")]),
 ("mesocortical","Mesocortical","Executive control, working memory",[("VTA","DLPFC"),("VTA","ACC"),("VTA","OFC")]),
 ("nigrostriatal","Nigrostriatal","Movement, habit learning",[("SNc","PUT"),("SNc","CAU")]),
 ("tubero","Tuberoinfundibular","Prolactin control",[("HYP","PIT")]),
 ("direct","Direct loop (D1)","Striatum to pallidum to thalamus: facilitates action",[("PUT","GP"),("CAU","GP"),("GP","THA"),("THA","DLPFC")]),
 ("indirect","Indirect loop (D2)","Striatum via subthalamic nucleus: suppresses action",[("PUT","GP"),("GP","STN"),("STN","GP"),("GP","THA")]),
]
