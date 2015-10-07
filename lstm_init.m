function nn = lstm_init(In,Hidden,Out)
k = 3*Hidden;
q=0.1;
nn.Wx = randn(In,Hidden) * q;
nn.bx = zeros(1,Hidden);
nn.Wxv = zeros(In,Hidden); % velocity, for applying momentum
nn.bxv = zeros(1,Hidden);

nn.Wf = randn(k,Hidden) * q;
nn.bf = zeros(1,Hidden);
nn.Wfv = zeros(k,Hidden); % velocity, for applying momentum
nn.bfv = zeros(1,Hidden);

nn.Wi = randn(k,Hidden) * q;
nn.bi = zeros(1,Hidden);
nn.Wiv = zeros(k,Hidden);
nn.biv = zeros(1,Hidden);

nn.Wo = randn(k,Hidden) * q;
nn.bo = zeros(1,Hidden);
nn.Wov = zeros(k,Hidden);
nn.bov = zeros(1,Hidden);

nn.Wc = randn(2*Hidden,Hidden) * q;
nn.bc = zeros(1,Hidden);
nn.Wcv = zeros(2*Hidden,Hidden);
nn.bcv = zeros(1,Hidden);

nn.Wout = randn(Hidden,Out) * q;
nn.bout = zeros(1,Out);
nn.Woutv = zeros(Hidden,Out);
nn.boutv = zeros(1,Out);

nn.widthOut = Out;
nn.widthIn = In;
nn.widthHidden = Hidden;

% initialise the nonlinearities inside the lstm
% to use different nonlinearities, change them here.
nn.sigm = @(x)logsig(x);
nn.dsigm = @(x)x.*(1-x);
nn.tanh = @(x)tansig(x);
nn.dtanh = @(x)(1-x.^2);

% nn.sigm = @(x)max(0,min(1,x));
% nn.dsigm = @(x) (x>0)&(x<1);
% nn.tanh = @(x)max(0,x);
% nn.dtanh = @(x)x>0;

