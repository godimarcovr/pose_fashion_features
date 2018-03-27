function D2=tmpdistfun_fn(ZI,ZJ)
    global tmp_custom_distance
    D2 = 1.0 - tmp_custom_distance(ZI,ZJ);
    D2 = D2';
end