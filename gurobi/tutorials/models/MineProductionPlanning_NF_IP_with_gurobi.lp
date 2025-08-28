\ Model MineProductionPlanningNF-IP
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x48d94b4ea20642a6
Maximize
  - 18.18181818181818 y[1,1] - 16.52892561983471 y[1,2]
   - 15.02629601803155 y[1,3] - 18.18181818181818 y[2,1]
   - 16.52892561983471 y[2,2] - 15.02629601803155 y[2,3]
   - 27.27272727272727 y[3,1] - 24.79338842975206 y[3,2]
   - 22.53944402704733 y[3,3] - 18.18181818181818 y[4,1]
   - 16.52892561983471 y[4,2] - 15.02629601803155 y[4,3]
   - 14.54545454545454 y[5,1] - 13.22314049586777 y[5,2]
   - 12.02103681442524 y[5,3] + 681.8181818181818 z_p[1,1,P]
   + 619.8347107438016 z_p[1,2,P] + 563.4861006761831 z_p[1,3,P]
   + 500 z_p[2,1,P] + 454.5454545454545 z_p[2,2,P]
   + 413.2231404958677 z_p[2,3,P] + 340.9090909090909 z_p[3,1,P]
   + 309.9173553719008 z_p[3,2,P] + 281.7430503380916 z_p[3,3,P]
   + 409.0909090909091 z_p[4,1,P] + 371.9008264462809 z_p[4,2,P]
   + 338.0916604057099 z_p[4,3,P] + 109.0909090909091 z_p[5,1,P]
   + 99.17355371900825 z_p[5,2,P] + 90.15777610818931 z_p[5,3,P]
   - 4.545454545454545 oreSP[P,S,1] - 4.132231404958677 oreSP[P,S,2]
   - 3.756574004507888 oreSP[P,S,3] + 90.90909090909091 metalSP[P,S,1]
   + 82.64462809917354 metalSP[P,S,2] + 75.13148009015775 metalSP[P,S,3]
Subject To
 cumulative[1,2]: x[1,1] - x[1,2] <= 0
 cumulative[1,3]: x[1,2] - x[1,3] <= 0
 cumulative[2,2]: x[2,1] - x[2,2] <= 0
 cumulative[2,3]: x[2,2] - x[2,3] <= 0
 cumulative[3,2]: x[3,1] - x[3,2] <= 0
 cumulative[3,3]: x[3,2] - x[3,3] <= 0
 cumulative[4,2]: x[4,1] - x[4,2] <= 0
 cumulative[4,3]: x[4,2] - x[4,3] <= 0
 cumulative[5,2]: x[5,1] - x[5,2] <= 0
 cumulative[5,3]: x[5,2] - x[5,3] <= 0
 link_xy[1,1]: x[1,1] - y[1,1] <= 0
 link_xy[1,2]: x[1,2] - y[1,1] - y[1,2] <= 0
 link_xy[1,3]: x[1,3] - y[1,1] - y[1,2] - y[1,3] <= 0
 link_xy[2,1]: x[2,1] - y[2,1] <= 0
 link_xy[2,2]: x[2,2] - y[2,1] - y[2,2] <= 0
 link_xy[2,3]: x[2,3] - y[2,1] - y[2,2] - y[2,3] <= 0
 link_xy[3,1]: x[3,1] - y[3,1] <= 0
 link_xy[3,2]: x[3,2] - y[3,1] - y[3,2] <= 0
 link_xy[3,3]: x[3,3] - y[3,1] - y[3,2] - y[3,3] <= 0
 link_xy[4,1]: x[4,1] - y[4,1] <= 0
 link_xy[4,2]: x[4,2] - y[4,1] - y[4,2] <= 0
 link_xy[4,3]: x[4,3] - y[4,1] - y[4,2] - y[4,3] <= 0
 link_xy[5,1]: x[5,1] - y[5,1] <= 0
 link_xy[5,2]: x[5,2] - y[5,1] - y[5,2] <= 0
 link_xy[5,3]: x[5,3] - y[5,1] - y[5,2] - y[5,3] <= 0
 frac_leq1[1]: y[1,1] + y[1,2] + y[1,3] <= 1
 frac_leq1[2]: y[2,1] + y[2,2] + y[2,3] <= 1
 frac_leq1[3]: y[3,1] + y[3,2] + y[3,3] <= 1
 frac_leq1[4]: y[4,1] + y[4,2] + y[4,3] <= 1
 frac_leq1[5]: y[5,1] + y[5,2] + y[5,3] <= 1
 precedence[3,1,1]: - x[1,1] + y[3,1] <= 0
 precedence[3,1,2]: - x[1,2] + y[3,1] + y[3,2] <= 0
 precedence[3,1,3]: - x[1,3] + y[3,1] + y[3,2] + y[3,3] <= 0
 precedence[3,2,1]: - x[2,1] + y[3,1] <= 0
 precedence[3,2,2]: - x[2,2] + y[3,1] + y[3,2] <= 0
 precedence[3,2,3]: - x[2,3] + y[3,1] + y[3,2] + y[3,3] <= 0
 precedence[4,3,1]: - x[3,1] + y[4,1] <= 0
 precedence[4,3,2]: - x[3,2] + y[4,1] + y[4,2] <= 0
 precedence[4,3,3]: - x[3,3] + y[4,1] + y[4,2] + y[4,3] <= 0
 precedence[5,3,1]: - x[3,1] + y[5,1] <= 0
 precedence[5,3,2]: - x[3,2] + y[5,1] + y[5,2] <= 0
 precedence[5,3,3]: - x[3,3] + y[5,1] + y[5,2] + y[5,3] <= 0
 precedence[5,4,1]: - x[4,1] + y[5,1] <= 0
 precedence[5,4,2]: - x[4,2] + y[5,1] + y[5,2] <= 0
 precedence[5,4,3]: - x[4,3] + y[5,1] + y[5,2] + y[5,3] <= 0
 split_mined_to_routing[1,1]: - y[1,1] + z_p[1,1,P] + z_s[1,1,S] <= 0
 split_mined_to_routing[1,2]: - y[1,2] + z_p[1,2,P] + z_s[1,2,S] <= 0
 split_mined_to_routing[1,3]: - y[1,3] + z_p[1,3,P] + z_s[1,3,S] <= 0
 split_mined_to_routing[2,1]: - y[2,1] + z_p[2,1,P] + z_s[2,1,S] <= 0
 split_mined_to_routing[2,2]: - y[2,2] + z_p[2,2,P] + z_s[2,2,S] <= 0
 split_mined_to_routing[2,3]: - y[2,3] + z_p[2,3,P] + z_s[2,3,S] <= 0
 split_mined_to_routing[3,1]: - y[3,1] + z_p[3,1,P] + z_s[3,1,S] <= 0
 split_mined_to_routing[3,2]: - y[3,2] + z_p[3,2,P] + z_s[3,2,S] <= 0
 split_mined_to_routing[3,3]: - y[3,3] + z_p[3,3,P] + z_s[3,3,S] <= 0
 split_mined_to_routing[4,1]: - y[4,1] + z_p[4,1,P] + z_s[4,1,S] <= 0
 split_mined_to_routing[4,2]: - y[4,2] + z_p[4,2,P] + z_s[4,2,S] <= 0
 split_mined_to_routing[4,3]: - y[4,3] + z_p[4,3,P] + z_s[4,3,S] <= 0
 split_mined_to_routing[5,1]: - y[5,1] + z_p[5,1,P] + z_s[5,1,S] <= 0
 split_mined_to_routing[5,2]: - y[5,2] + z_p[5,2,P] + z_s[5,2,S] <= 0
 split_mined_to_routing[5,3]: - y[5,3] + z_p[5,3,P] + z_s[5,3,S] <= 0
 sp_bal_ore[S,2]: 10 z_s[1,1,S] + 10 z_s[2,1,S] + 15 z_s[3,1,S]
   + 10 z_s[4,1,S] + 8 z_s[5,1,S] - oreSP[P,S,2] + oreS_rem[S,1]
   - oreS_rem[S,2] = 0
 sp_bal_ore[S,3]: 10 z_s[1,2,S] + 10 z_s[2,2,S] + 15 z_s[3,2,S]
   + 10 z_s[4,2,S] + 8 z_s[5,2,S] - oreSP[P,S,3] + oreS_rem[S,2]
   - oreS_rem[S,3] = 0
 sp_bal_metal[S,2]: 8 z_s[1,1,S] + 6 z_s[2,1,S] + 4.5 z_s[3,1,S]
   + 5 z_s[4,1,S] + 1.6 z_s[5,1,S] - metalSP[P,S,2] + metalS_rem[S,1]
   - metalS_rem[S,2] = 0
 sp_bal_metal[S,3]: 8 z_s[1,2,S] + 6 z_s[2,2,S] + 4.5 z_s[3,2,S]
   + 5 z_s[4,2,S] + 1.6 z_s[5,2,S] - metalSP[P,S,3] + metalS_rem[S,2]
   - metalS_rem[S,3] = 0
 sp_init_ore[S]: oreS_rem[S,1] = 0
 sp_end_ore[S]: oreS_rem[S,3] = 0
 sp_init_metal[S]: metalS_rem[S,1] = 0
 sp_end_metal[S]: metalS_rem[S,3] = 0
 no_reclaim_ore_t1[S]: oreSP[P,S,1] = 0
 no_reclaim_metal_t1[S]: metalSP[P,S,1] = 0
 mining_cap[1]: 10 y[1,1] + 10 y[2,1] + 15 y[3,1] + 10 y[4,1] + 8 y[5,1]
   <= 15
 mining_cap[2]: 10 y[1,2] + 10 y[2,2] + 15 y[3,2] + 10 y[4,2] + 8 y[5,2]
   <= 15
 mining_cap[3]: 10 y[1,3] + 10 y[2,3] + 15 y[3,3] + 10 y[4,3] + 8 y[5,3]
   <= 15
 processing_cap[1]: 10 z_p[1,1,P] + 10 z_p[2,1,P] + 15 z_p[3,1,P]
   + 10 z_p[4,1,P] + 8 z_p[5,1,P] + oreSP[P,S,1] <= 10
 processing_cap[2]: 10 z_p[1,2,P] + 10 z_p[2,2,P] + 15 z_p[3,2,P]
   + 10 z_p[4,2,P] + 8 z_p[5,2,P] + oreSP[P,S,2] <= 10
 processing_cap[3]: 10 z_p[1,3,P] + 10 z_p[2,3,P] + 15 z_p[3,3,P]
   + 10 z_p[4,3,P] + 8 z_p[5,3,P] + oreSP[P,S,3] <= 10
Bounds
 y[1,1] <= 1
 y[1,2] <= 1
 y[1,3] <= 1
 y[2,1] <= 1
 y[2,2] <= 1
 y[2,3] <= 1
 y[3,1] <= 1
 y[3,2] <= 1
 y[3,3] <= 1
 y[4,1] <= 1
 y[4,2] <= 1
 y[4,3] <= 1
 y[5,1] <= 1
 y[5,2] <= 1
 y[5,3] <= 1
 z_p[1,1,P] <= 1
 z_p[1,2,P] <= 1
 z_p[1,3,P] <= 1
 z_p[2,1,P] <= 1
 z_p[2,2,P] <= 1
 z_p[2,3,P] <= 1
 z_p[3,1,P] <= 1
 z_p[3,2,P] <= 1
 z_p[3,3,P] <= 1
 z_p[4,1,P] <= 1
 z_p[4,2,P] <= 1
 z_p[4,3,P] <= 1
 z_p[5,1,P] <= 1
 z_p[5,2,P] <= 1
 z_p[5,3,P] <= 1
 z_s[1,1,S] <= 1
 z_s[1,2,S] <= 1
 z_s[1,3,S] <= 1
 z_s[2,1,S] <= 1
 z_s[2,2,S] <= 1
 z_s[2,3,S] <= 1
 z_s[3,1,S] <= 1
 z_s[3,2,S] <= 1
 z_s[3,3,S] <= 1
 z_s[4,1,S] <= 1
 z_s[4,2,S] <= 1
 z_s[4,3,S] <= 1
 z_s[5,1,S] <= 1
 z_s[5,2,S] <= 1
 z_s[5,3,S] <= 1
Binaries
 x[1,1] x[1,2] x[1,3] x[2,1] x[2,2] x[2,3] x[3,1] x[3,2] x[3,3] x[4,1]
 x[4,2] x[4,3] x[5,1] x[5,2] x[5,3]
End
