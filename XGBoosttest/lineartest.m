x = 0:1:1000;
x = x + randn([1,1001]);
a = rand();
b = rand()/rand();
y = a * x + b + randn([1,1001]);

plot(x,y)
data = [x; y]';
%%
csvwrite('ltest.csv', data)