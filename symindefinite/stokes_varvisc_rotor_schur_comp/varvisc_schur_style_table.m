function sty = varvisc_schur_style_table(n)
%VARVISC_SCHUR_STYLE_TABLE  Colourblind-safe curve styles.
    colors = [0 0 0; .84 .37 0; 0 .45 .70; 0 .62 .45; .80 .47 .65; .6 .6 .6];
    markers = {'o','s','^','d','v','p'};
    lines = {'-','--','-.','-',':','--'};
    sty = repmat(struct('color',[0 0 0],'marker','o','linestyle','-', ...
                        'linewidth',1.6),n,1);
    for i = 1:n
        j = mod(i-1,size(colors,1))+1;
        sty(i).color = colors(j,:); sty(i).marker = markers{j};
        sty(i).linestyle = lines{j}; sty(i).linewidth = 2.2-.15*min(i-1,6);
    end
end
