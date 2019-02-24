clear;
clc;
figure('Name','Figure 1', 'units','inch','position',[1.5,1.5,12,8]);
data=readtable('CC3.SI.csv');
data(data.Volume==0,:)=[];
X=datetime(data.Date);
Y=data.AdjClose;
plot(X,Y,'k-','LineWidth',1);
hold on;
ave15=round(movmean(Y,15,'Endpoints','discard'),3);
ave15(1:35)=[];
ave50=round(movmean(Y,50,'Endpoints','discard'),3);
daxis=X(50:end);
paxis=Y(50:end);
plot(daxis,ave15,'b-');
plot(daxis,ave50,'c-');
x=ave15-ave50;
x(x>0)=1;
x(x<=0)=0;
y=diff(x); %size is reduced by 1
idxSell=find(y<0)+1;
idxBuy=find(y>0)+1;
plot(daxis(idxBuy),paxis(idxBuy), ... 
     'y.','MarkerSize',20,'Linewidth',1);
plot(daxis(idxSell),paxis(idxSell), ... 
     'r.','MarkerSize',20,'Linewidth',1);
legend('Adj Close', '15d', '50d', 'crossSell', 'crossBuy');
xlabel('Date');
axis tight
set(gca,'XTickLabelRotation',30)