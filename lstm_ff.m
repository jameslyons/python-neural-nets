function nnres = lstm_ff(seq,nn)
% the forward pass of an lstm RNN
L = size(seq,1);
%x = zeros(L,nn.widthOut);
f = zeros(L,nn.widthHidden);
i = zeros(L,nn.widthHidden);
o = zeros(L,nn.widthHidden);
Chat = zeros(L,nn.widthHidden);
C = zeros(L,nn.widthHidden);
h = zeros(L,nn.widthHidden);
%out = zeros(L,nn.widthOut);
sigm = nn.sigm;
tanh = nn.tanh;

% pass input through first non-recurrent weight layer
x = nn.tanh(seq*nn.Wx + repmat(nn.bx,L,1));
%intialise everything for first time step
f(1,:) = sigm([zeros(1,nn.widthHidden),x(1,:),zeros(1,nn.widthHidden)]*nn.Wf + nn.bf);
i(1,:) = sigm([zeros(1,nn.widthHidden),x(1,:),zeros(1,nn.widthHidden)]*nn.Wi + nn.bi);
Chat(1,:) = tanh([zeros(1,nn.widthHidden),x(1,:)]*nn.Wc + nn.bc);
C(1,:) = i(1,:).*Chat(1,:);
o(1,:) = sigm([zeros(1,nn.widthHidden),x(1,:),C(1,:)]*nn.Wo + nn.bo);
h(1,:) = o(1,:).*tanh(C(1,:));
% loop over all the other time steps
for t = 2:L
    f(t,:) = sigm([h(t-1,:),x(t,:),C(t-1,:)]*nn.Wf + nn.bf);
    i(t,:) = sigm([h(t-1,:),x(t,:),C(t-1,:)]*nn.Wi + nn.bi);
    Chat(t,:) = tanh([h(t-1,:),x(t,:)]*nn.Wc + nn.bc);
    C(t,:) = f(t,:).*C(t-1,:) + i(t,:).*Chat(t,:);
    o(t,:) = sigm([h(t-1,:),x(t,:),C(t,:)]*nn.Wo + nn.bo);
    h(t,:) = o(t,:).*tanh(C(t,:));
end
% pass output through non-recurrent transformation layer
out = nn.tanh(h*nn.Wout + repmat(nn.bout,L,1));
%out = h;

nnres.x = x;
nnres.f = f;
nnres.i = i;
nnres.Chat = Chat;
nnres.C = C;
nnres.o = o;
nnres.h = h;
nnres.out = out;
nnres.in = seq;
