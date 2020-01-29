# GPU-query
lists GPU parameters & settings for persistent kernel

Queries / prints GPU parameters and full occupancy kernel settings if maximum flexibility of warp scheduling is targeted.
- maximum number of active warps
- maximum number of scheduled, but inactive warps 
   all warps are loaded to SM, if memory transaction doesn't allow execution of an active warp, inactive ones become active (swapping). 

