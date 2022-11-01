f = dir('*.png');
N=20;
IMG = zeros([500 1000 3 N]);
for k=0:N:length(f)
    montage(IMG/255,'Size', [4 5])
    title(k)
    drawnow
    for i=1:N
        IMG(:,:,:,i)=imread(f(i+k).name);
    end
    pause(2)
end

