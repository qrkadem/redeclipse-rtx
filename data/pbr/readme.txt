Physically based replacements for the most used legacy surfaces.

All materials are from Poly Haven (https://polyhaven.com) or ambientCG (https://ambientcg.com) and are CC0: public domain,
no attribution required. They are credited here anyway. Maps are the 2k JPGs; height.png is the displacement stretched to
its full range, and arm.jpg packs ambient occlusion, roughness and metalness into r, g, b.

asphalt_02                 Asphalt 02                   3.0 m  by Rob Tuytel  https://polyhaven.com/a/asphalt_02
concrete_block_wall        Concrete Block Wall          2.0 m  by Charlotte Baglioni  https://polyhaven.com/a/concrete_block_wall
concrete_floor_worn_001    Concrete Floor Worn 001      3.0 m  by Dimitrios Savva, Rico Cilliers  https://polyhaven.com/a/concrete_floor_worn_001
concrete_wall_006          Concrete Wall 006            2.0 m  by Dario Barresi, Charlotte Baglioni  https://polyhaven.com/a/concrete_wall_006
dense_sand                 Dense Sand                   1.8 m  by Dimitrios Savva  https://polyhaven.com/a/dense_sand
plastered_wall             Plastered Wall               2.0 m  by Amal Kumar  https://polyhaven.com/a/plastered_wall
red_laterite_soil_stones   Red Laterite Soil Stones     2.0 m  by Amal Kumar  https://polyhaven.com/a/red_laterite_soil_stones
rock_face_04               Rock Face 04                 2.0 m  by Dimitrios Savva  https://polyhaven.com/a/rock_face_04
Metal038                   Metal 038                    by ambientCG  https://ambientcg.com/a/Metal038
Metal038_holes             Metal038 with the dimple layout of trak5/holes1a regenerated over it (tools/make_holes.py)

tools/ has the scripts that fetched and packed these: fetch_ph.py <dest> <ids> for Poly Haven, fetch_acg.py <dest> <tmp> <ids>
for ambientCG, and make_holes.py <Metal038 dir> <dest>. Each replacement's diffuse was then scaled to the mean colour of the
texture it replaces (except the metals, see package.cfg).
