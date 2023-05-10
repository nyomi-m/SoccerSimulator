%File for building K matrices from outputs
K_New = [2935.98519 0 2072.91322; 0 2865.46075 1591.44911; 0 0 1];
K_Old = [1713.22736 0 2072.91322; 0 1721.16188 1591.44911; 0 0 1];

save('KMatirces', 'K_Old', 'K_New');