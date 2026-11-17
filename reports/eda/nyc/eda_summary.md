# EDA: Foursquare TSMC2014 NYC

## 1. Data integrity

### Overview

```
check-ins                                               227428
users                                                     1083
venues                                                   38333
category IDs                                               400
category names                                             251
first check-in (local)                     2012-04-03 14:00:09
last check-in (local)                      2013-02-15 21:35:36
duplicate (user, venue, timestamp) rows                    250
distinct user-venue pairs                                91024
user-venue density (%)                                  0.2193
repeat check-ins (%)                                     59.98
```

### Category join coverage

```
check-ins with a path (%)      99.975
category IDs without a path     1.000
check-ins without a path       56.000
```

### Category IDs without a path

```
                 categoryId category_name  check-ins
0  4e51a0c0bd41d3446defbb2e         Ferry         56
```

### Venues with several category IDs

The venue table keeps each venue's most frequent category ID.

```
venues with > 1 category ID    337.00
share of venues (%)              0.88
share of check-ins (%)           3.18
```

### Examples

```
                    venueId                                     category names
0  3fd66200f964a52001e81ee3                                                Bar
1  3fd66200f964a52054e91ee3               American Restaurant | Breakfast Spot
2  3fd66200f964a52057e31ee3                     French Restaurant | Restaurant
3  3fd66200f964a5205fe41ee3               Department Store | Electronics Store
4  3fd66200f964a5207cf11ee3                        American Restaurant | Diner
5  3fd66200f964a52083e61ee3                     French Restaurant | Restaurant
6  3fd66200f964a520a0e81ee3                                                Bar
7  3fd66200f964a520b1e91ee3  Korean Restaurant | Vegetarian / Vegan Restaurant
8  3fd66200f964a520b2e71ee3                                                Bar
9  3fd66200f964a520b7e31ee3                                                Bar
```

### Venue coordinate spread

The venue table uses the median coordinate.

```
venues with > 1 coordinate     3484.0
median spread of those (m)       24.4
p90 spread (m)                  139.9
max spread (m)                20875.2
venues with spread > 100 m      463.0
venues with spread > 1 km        56.0
```

### UTC offsets (minutes)

```
                check-ins  share (%)
timezoneOffset                      
-240               170569     74.999
-300                56403     24.800
 540                  407      0.179
 420                   30      0.013
-420                    8      0.004
 120                    4      0.002
-360                    4      0.002
 480                    1      0.000
-180                    1      0.000
 660                    1      0.000
```

## 2. Interaction sparsity

### Per-user activity

```
                count   mean    std    min    10%    50%    90%    99%     max
checkins       1083.0  210.0  188.5  100.0  107.0  153.0  362.4  902.3  2697.0
distinct_pois  1083.0   84.0   47.9    9.0   41.0   76.0  133.8  228.9   714.0
```

### Users affected by TEMN settings

```
users with < 10 distinct POIs (dropped by paper filter)            1
users with > 100 distinct POIs (history cut at max_p_num=100)    295
users with > 200 distinct POIs (history cut at max_p_num=200)     23
```

### Venue popularity

```
venues with 1 check-in (%)            43.300
venues with 1 user (%)                63.500
venues with >= 5 users (%)            10.200
check-ins at top 1% of venues (%)     19.100
check-ins at top 10% of venues (%)    58.800
Gini of check-ins per venue            0.675
```

## 3. Chronological split (paper protocol)

### Users and pairs per split

```
       users  pairs
split              
test    1082  13571
train   1082  63751
val     1082  13693
```

### Cold POIs among held-out POIs

```
       held-out user-POI pairs  distinct POIs  pairs with a cold POI (%)  distinct cold POIs
split                                                                                       
val                      13693           9821                       32.5                4199
test                     13571           9522                       33.7                4166
```

### Category headroom: is the test POI's category already in the user's history?

```
            distinct nodes  test POI category in user's train history (%)  random unvisited POI (%)  lift
path level                                                                                               
level 1                 10                                           97.3                      93.9  1.04
level 2                175                                           74.8                      57.9  1.29
level 3                328                                           54.1                      34.1  1.59
leaf                   397                                           50.2                      30.4  1.65
```

### Unseen categories in test

```
test pairs whose leaf category has no train POI (%)    0.07
distinct unseen leaf categories in test                5.00
```

## 4. Time

### TLDA time slots

```
slots (24 hours x weekday/weekend)       48.0
min check-ins per slot                  636.0
median check-ins per slot              3862.0
max check-ins per slot                12612.0
weekend share of check-ins (%)           27.6
```

## 5. Category paths

### Depth: taxonomy vs data

```
       categories in taxonomy  categories used  venues  check-ins  check-ins (%)
depth                                                                           
1                          11                7     186        649            0.3
2                         457              141   13656      78700           34.6
3                         474              173   19854     127944           56.3
4                         241               71    4281      19001            8.4
5                          82                5     346       1075            0.5
6                          14                0       0          0            0.0
```

### Granularity when cutting paths at each level

```
       distinct nodes  effective number  top node share of check-ins (%)  median venues per node  nodes with < 10 venues
level                                                                                                                   
1                  10               7.2                             29.6                    3307                       0
2                 175              36.5                             15.5                      44                      37
3                 328             106.9                              6.7                      43                      74
4                 392             125.1                              6.7                      39                      93
5                 397             125.9                              6.7                      38                      93
6                 397             125.9                              6.7                      38                      93
```

### Category names vs IDs

```
category names                     251.0
category IDs                       400.0
names with > 1 ID                   32.0
check-ins under those names (%)     49.8
```

### Names with several IDs

```
                           ids  paths  checkins                                                                                                                                                                                                                                                                        path_examples
category_name                                                                                                                                                                                                                                                                                                                       
College Academic Building   19     19      3479  Community and Government > Education > College and University > College Academic Building | Community and Government > Education > College and University > College Administrative Building | Community and Government > Education > College and University > College Arts Building
Bar                         16     16     15978                                                                                                                                                                                Arts and Entertainment > Night Club | Arts and Entertainment > Strip Club | Dining and Drinking > Bar
Other Great Outdoors        13     13      4134                                                                                                                                                                                      Dining and Drinking > Vineyard | Landmarks and Outdoors > Farm | Landmarks and Outdoors > Field
Food & Drink Shop           11     11      6596                                                                                                                                                                   Dining and Drinking > Food Court | Retail > Food and Beverage Retail | Retail > Food and Beverage Retail > Butcher
Athletic & Sport             9      9       703                                                                                                                                                    Sports and Recreation | Sports and Recreation > Baseball > Baseball Field | Sports and Recreation > Basketball > Basketball Court
Medical Center               8      8      3366                                                                                                                                                            Health and Medicine > Dentist | Health and Medicine > Emergency Service > Emergency Room | Health and Medicine > Hospital
Clothing Store               8      8      2976                                                                                                                                                                   Retail > Boutique | Retail > Fashion Retail > Children's Clothing Store | Retail > Fashion Retail > Clothing Store
Government Building          8      8      1589                                                                                                      Community and Government > Government Building | Community and Government > Government Building > Capitol Building | Community and Government > Government Building > City Hall
Stadium                      8      8      1120                                                                                                                                       Arts and Entertainment > Stadium | Arts and Entertainment > Stadium > Baseball Stadium | Arts and Entertainment > Stadium > Basketball Stadium
Office                       7      7     12740                                                                                                             Business and Professional Services > Office | Business and Professional Services > Office > Coworking Space | Business and Professional Services > Office > Meeting Room
Gym / Fitness Center         7      7      9171                                                                                                                                        Sports and Recreation > Gym and Studio | Sports and Recreation > Gym and Studio > Climbing Gym | Sports and Recreation > Gym and Studio > Gym
Hotel                        7      7      2860                                                                                                                                                 Dining and Drinking > Bar > Hotel Bar | Landmarks and Outdoors > Roof Deck | Travel and Transportation > Lodging > Bed and Breakfast
Airport                      7      7      2852                                                                                  Travel and Transportation > Transport Hub > Airport | Travel and Transportation > Transport Hub > Airport > Airport Food Court | Travel and Transportation > Transport Hub > Airport > Airport Gate
Ski Area                     6      6        32                                                                                                                                       Dining and Drinking > Bar > Apres Ski Bar | Sports and Recreation > Snow Sports > Ski Chalet | Sports and Recreation > Snow Sports > Ski Lodge
College Stadium              5      5        75         Community and Government > Education > College and University > College Baseball Diamond | Community and Government > Education > College and University > College Basketball Court | Community and Government > Education > College and University > College Football Field
```

### Frequency buckets (RQ3)

```
                    categories  venues  checkins  check-ins (%)
bucket                                                         
rare (< 10 venues)          93     452      2862            1.3
medium (10-99)             200    7898     45481           20.0
frequent (>= 100)          104   29973    179026           78.7
```

### Top leaf categories

```
                                                      full_category_path  venues  users  checkins  check-ins (%)
0       Community and Government > Residential Building > Home (private)    1270    585     15281           6.72
1                            Business and Professional Services > Office     984    627      9621           4.23
2              Travel and Transportation > Transport Hub > Metro Station     410    370      9480           4.17
3        Dining and Drinking > Cafe, Coffee, and Tea House > Coffee Shop     841    799      7481           3.29
4                                              Dining and Drinking > Bar     918    749      6179           2.72
5               Travel and Transportation > Transport Hub > Rail Station     172    580      5362           2.36
6                           Sports and Recreation > Gym and Studio > Gym     340    471      5075           2.23
7      Landmarks and Outdoors > States and Municipalities > Neighborhood     297    475      4720           2.08
8                                          Landmarks and Outdoors > Park     501    755      4716           2.07
9                      Retail > Food and Beverage Retail > Grocery Store     679    747      4342           1.91
10                               Dining and Drinking > Restaurant > Deli     702    509      4257           1.87
11  Community and Government > Residential Building > Apartment or Condo     655    404      4228           1.86
12                                    Landmarks and Outdoors > Structure     679    536      3449           1.52
13                Dining and Drinking > Restaurant > American Restaurant     609    787      3254           1.43
14                                      Travel and Transportation > Road     376    350      3202           1.41
15                                Sports and Recreation > Gym and Studio     142    232      2794           1.23
16                                                     Retail > Pharmacy     604    495      2488           1.09
17                           Travel and Transportation > Lodging > Hotel     249    434      2487           1.09
18                           Dining and Drinking > Restaurant > Pizzeria     842    712      2470           1.09
19                         Landmarks and Outdoors > Other Great Outdoors     415    552      2418           1.06
```

### Top-level categories

```
                                    categories  venues  checkins  venues (%)  check-ins (%)
level1_category_name                                                                       
Dining and Drinking                        106   13902     67302        36.3           29.6
Travel and Transportation                   34    3835     34408        10.0           15.1
Community and Government                    65    4338     32445        11.3           14.3
Retail                                      60    5731     24205        15.0           10.6
Landmarks and Outdoors                      31    3094     23968         8.1           10.5
Business and Professional Services          27    3520     19437         9.2            8.5
Arts and Entertainment                      42    1995     11670         5.2            5.1
Sports and Recreation                       23     997     10305         2.6            4.5
Health and Medicine                          7     843      3341         2.2            1.5
Nightlife Spot                               2      68       288         0.2            0.1
```

### Per-user category diversity

```
                              count   mean    std   min    10%    50%    90%     max
distinct leaf categories     1083.0  47.52  17.62  4.00  28.00  46.00  69.00  180.00
distinct level-1 categories  1083.0   7.89   1.09  3.00   7.00   8.00   9.00   10.00
top category share (%)       1083.0  22.74  12.43  4.72   9.82  20.25  38.23   91.43
normalised entropy           1083.0   0.80   0.10  0.22   0.67   0.81   0.91    0.97
```

## 6. Category embeddings

### Coverage

```
categories used by venues     398
of those with an embedding    397
embedding dimension           384
```

### Taxonomy separation (cosine)

```
mean cos, same level-1         0.626
mean cos, different level-1    0.263
```

### Per top-level category

```
                                    categories  mean cos within  mean cos to others    gap
level-1 category                                                                          
Business and Professional Services          27            0.453               0.263  0.190
Travel and Transportation                   34            0.577               0.280  0.297
Community and Government                    65            0.510               0.209  0.301
Arts and Entertainment                      42            0.608               0.304  0.304
Sports and Recreation                       23            0.641               0.313  0.328
Health and Medicine                          7            0.645               0.259  0.386
Retail                                      60            0.643               0.251  0.392
Dining and Drinking                        106            0.676               0.281  0.395
Landmarks and Outdoors                      31            0.684               0.233  0.451
Nightlife Spot                               2            0.938               0.236  0.702
```

### Nearest neighbours

```
                                                               query  rank                                                             neighbour  cosine
0                               Dining and Drinking > Bar > Sake Bar     1                                             Dining and Drinking > Bar   0.865
1                               Dining and Drinking > Bar > Sake Bar     2                                  Dining and Drinking > Bar > Wine Bar   0.858
2                               Dining and Drinking > Bar > Sake Bar     3                              Dining and Drinking > Bar > Cocktail Bar   0.846
3                               Dining and Drinking > Bar > Sake Bar     4                                Dining and Drinking > Bar > Whisky Bar   0.842
4                               Dining and Drinking > Bar > Sake Bar     5                                 Dining and Drinking > Bar > Hotel Bar   0.841
5   Community and Government > Residential Building > Home (private)     1  Community and Government > Residential Building > Apartment or Condo   0.837
6   Community and Government > Residential Building > Home (private)     2                        Community and Government > Government Building   0.779
7   Community and Government > Residential Building > Home (private)     3                        Community and Government > Housing Development   0.755
8   Community and Government > Residential Building > Home (private)     4            Community and Government > Government Building > City Hall   0.734
9   Community and Government > Residential Building > Home (private)     5     Community and Government > Government Building > Capitol Building   0.711
10                      Sports and Recreation > Gym and Studio > Gym     1                                Sports and Recreation > Gym and Studio   0.980
11                      Sports and Recreation > Gym and Studio > Gym     2                 Sports and Recreation > Gym and Studio > Dance Studio   0.916
12                      Sports and Recreation > Gym and Studio > Gym     3                     Sports and Recreation > Gym and Studio > Gym Pool   0.893
13                      Sports and Recreation > Gym and Studio > Gym     4                  Sports and Recreation > Gym and Studio > Yoga Studio   0.861
14                      Sports and Recreation > Gym and Studio > Gym     5       Sports and Recreation > Water Sports > Swimming > Swimming Pool   0.743
15   Dining and Drinking > Cafe, Coffee, and Tea House > Coffee Shop     1              Dining and Drinking > Cafe, Coffee, and Tea House > Café   0.960
16   Dining and Drinking > Cafe, Coffee, and Tea House > Coffee Shop     2          Dining and Drinking > Cafe, Coffee, and Tea House > Tea Room   0.927
17   Dining and Drinking > Cafe, Coffee, and Tea House > Coffee Shop     3                                      Dining and Drinking > Restaurant   0.755
18   Dining and Drinking > Cafe, Coffee, and Tea House > Coffee Shop     4                         Dining and Drinking > Restaurant > Restaurant   0.755
19   Dining and Drinking > Cafe, Coffee, and Tea House > Coffee Shop     5                              Dining and Drinking > Restaurant > Diner   0.748
20                                   Landmarks and Outdoors > Bridge     1                                                Landmarks and Outdoors   0.786
21                                   Landmarks and Outdoors > Bridge     2                                    Landmarks and Outdoors > Structure   0.770
22                                   Landmarks and Outdoors > Bridge     3                                        Landmarks and Outdoors > River   0.751
23                                   Landmarks and Outdoors > Bridge     4                         Landmarks and Outdoors > Other Great Outdoors   0.742
24                                   Landmarks and Outdoors > Bridge     5                                         Landmarks and Outdoors > Park   0.740
```

### Near-duplicate embeddings (cosine >= 0.95)

```
                                                                                  category A                                                                                                       category B  cosine
0                 Community and Government > Education > College and University > University                                                    Community and Government > Education > College and University   0.983
1                                                     Sports and Recreation > Gym and Studio                                                                     Sports and Recreation > Gym and Studio > Gym   0.980
2                                                           Dining and Drinking > Restaurant                                                                    Dining and Drinking > Restaurant > Restaurant   0.973
3                      Sports and Recreation > Snow Sports > Ski Resort and Area > Ski Trail                                                        Sports and Recreation > Snow Sports > Ski Resort and Area   0.965
4          Community and Government > Education > College and University > Community College                                                    Community and Government > Education > College and University   0.963
5                                                   Dining and Drinking > Restaurant > Diner                                                                    Dining and Drinking > Restaurant > Restaurant   0.961
6                                   Dining and Drinking > Cafe, Coffee, and Tea House > Café                                                  Dining and Drinking > Cafe, Coffee, and Tea House > Coffee Shop   0.960
7                                             Arts and Entertainment > Performing Arts Venue                                                     Arts and Entertainment > Performing Arts Venue > Music Venue   0.959
8                               Dining and Drinking > Cafe, Coffee, and Tea House > Tea Room                                                         Dining and Drinking > Cafe, Coffee, and Tea House > Café   0.956
9                                                            Arts and Entertainment > Museum                                                                     Arts and Entertainment > Museum > Art Museum   0.956
10   Community and Government > Education > Primary and Secondary School > Elementary School                                Community and Government > Education > Primary and Secondary School > High School   0.953
11  Dining and Drinking > Restaurant > Latin American Restaurant > South American Restaurant  Dining and Drinking > Restaurant > Latin American Restaurant > South American Restaurant > Brazilian Restaurant   0.953
12                                                  Dining and Drinking > Restaurant > Diner                                                                                 Dining and Drinking > Restaurant   0.952
13                  Dining and Drinking > Restaurant > Asian Restaurant > Chinese Restaurant                                                              Dining and Drinking > Restaurant > Asian Restaurant   0.952
14          Dining and Drinking > Restaurant > American Restaurant > New American Restaurant                                                           Dining and Drinking > Restaurant > American Restaurant   0.952
15  Dining and Drinking > Restaurant > Latin American Restaurant > South American Restaurant                                                     Dining and Drinking > Restaurant > Latin American Restaurant   0.950
```

### Embedding similarity vs behaviour

```
categories compared                                297.000
category pairs                                   43956.000
rho(embedding cosine, behaviour)                     0.261
rho(same level-1, behaviour)                         0.172
rho(embedding, behaviour) within same level-1        0.332
rho(embedding, behaviour) across level-1             0.198
```
