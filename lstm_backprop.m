function nn = lstm_backprop(nn,input,labels,LR,momentum)
% function nn = lstm_backprop(nn,input,labels,LR,momentum)
% nn is the neural net created by lstm_init
% input is a N by D matrix (1 sequence). N is length of sequence
% labels is N by Q matrix of labels
nnres = lstm_ff(input,nn);
[L,W] = size(nnres.h); % our batch length
error = 2*(labels - nnres.out);

delta_o = zeros(L,W);
delta_x = zeros(L,W);
delta_f = zeros(L,W);
delta_i = zeros(L,W);
delta_C = zeros(L,W);
delta_Chat = zeros(L,W);
delta_C_next = zeros(1,W);
delta_h_next = zeros(1,W);

delta_out = error.*nn.dtanh(nnres.out);

Cm1 = [zeros(1,W);nnres.C(1:end-1,:)];
htm1 = [zeros(1,W);nnres.h(1:end-1,:)];
for k = L:-1:1
    err = delta_out(k,:)*nn.Wout' + delta_h_next;
    delta_o(k,:) = err.*nn.dsigm(nnres.o(k,:)).*nn.tanh(nnres.C(k,:));
    delta_C(k,:) = err.*nn.dtanh(nn.tanh(nnres.C(k,:))).*nnres.o(k,:) + delta_C_next;
    delta_f(k,:) = Cm1(k,:).*nn.dsigm(nnres.f(k,:)).*delta_C(k,:);
    delta_i(k,:) = nnres.Chat(k,:).*nn.dsigm(nnres.i(k,:)).*delta_C(k,:);
    delta_Chat(k,:) = nn.dtanh(nnres.Chat(k,:)).*nnres.i(k,:).*delta_C(k,:);
    
    delta_C_next = delta_C(k,:).*nnres.f(k,:);
    delta_h_next = delta_f(k,:)*nn.Wf(1:W,:)' + delta_i(k,:)*nn.Wi(1:W,:)' + delta_Chat(k,:)*nn.Wc(1:W,:)' + delta_o(k,:)*nn.Wo(1:W,:)';
    delta_x(k,:) = (delta_f(k,:)*nn.Wf(W+1:2*W,:)' + delta_i(k,:)*nn.Wi(W+1:2*W,:)' + ...
         delta_Chat(k,:)*nn.Wc(W+1:2*W,:)' + delta_o(k,:)*nn.Wo(W+1:2*W,:)').*nn.dtanh(nnres.x(k,:));
end

% all the deltas are calculated, now compute the updates
clip = 1; % if gradient > clip, set it to clip
grad = 1/L*nnres.h'*delta_out;
grad(grad > clip) = clip;
nn.Woutv = momentum*nn.Woutv + LR*grad;
nn.boutv = momentum*nn.boutv + 1/L*LR*sum(grad);
nn.Wout = nn.Wout + nn.Woutv;
nn.bout = nn.bout + nn.boutv;

grad = 1/L*[htm1,nnres.x,nnres.C]'*delta_o;
grad(grad > clip) = clip;
nn.Wov = momentum*nn.Wov + LR*grad;
nn.bov = momentum*nn.bov + 1/L*LR*sum(grad);
nn.Wo = nn.Wo + nn.Wov;
nn.bo = nn.bo + nn.bov;

grad = 1/L*[htm1,nnres.x,Cm1]'*delta_f;
grad(grad > clip) = clip;
nn.Wfv = momentum*nn.Wfv + LR*grad;
nn.bfv = momentum*nn.bfv + 1/L*LR*sum(grad);
nn.Wf = nn.Wf + nn.Wfv;
nn.bf = nn.bf + nn.bfv;

grad = 1/L*[htm1,nnres.x,Cm1]'*delta_i;
grad(grad > clip) = clip;
nn.Wiv = momentum*nn.Wiv + LR*grad;
nn.biv = momentum*nn.biv + 1/L*LR*sum(grad);
nn.Wi = nn.Wi + nn.Wiv;
nn.bi = nn.bi + nn.biv;

grad = 1/L*[htm1,nnres.x]'*delta_Chat;
grad(grad > clip) = clip;
nn.Wcv = momentum*nn.Wcv + LR*grad;
nn.bcv = momentum*nn.bcv + 1/L*LR*sum(grad);
nn.Wc = nn.Wc + nn.Wcv;
nn.bc = nn.bc + nn.bcv;

grad = 1/L*nnres.in'*delta_x;
grad(grad > clip) = clip;
nn.Wxv = momentum*nn.Wxv + LR*grad;
nn.bxv = momentum*nn.bxv + 1/L*LR*sum(grad);
nn.Wx = nn.Wx + nn.Wxv;
nn.bx = nn.bx + nn.bxv;
