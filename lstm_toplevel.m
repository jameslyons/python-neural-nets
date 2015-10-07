load combined_train_short.mat
seqs = combined;
load combined_test_short.mat
testseqs = combined;
LR = 0.1;
momentum = 0.9;
% get a randomly initialised lstm rnn
nn = lstm_init(20,35,4);
% train the network
for epochs = 1:30
    for i = 1:length(seqs)
        input = seqs(i).prob;
        label = id2oneofk2(seqs(i).ss,'CEHX'); % convert ss string to one hot 
        % process one sequence at a time, update weights after each seq
        nn = lstm_backprop(nn,input,label,LR,momentum);
        if mod(i,10)==0
            fprintf('.');
        end
    end
    fprintf('\n');
    % compute accuracy every epoch to see how good we are going
    correct = 0;
    total = 0;
    for i = 1:length(testseqs)
        input = testseqs(i).prob;
        tlabel = id2oneofk2(testseqs(i).ss,'CEHX');
        label = tlabel(:,1:3);
        
        nnres = lstm_ff(input,nn);
        [~,ind1] = max(nnres.out(:,1:3),[],2);
        [~,ind2] = max(label,[],2);
        ind1(tlabel(:,4)==1) = [];
        ind2(tlabel(:,4)==1) = [];
        
        correct = correct + sum(ind1==ind2);
        total = total + length(ind1);
        if mod(i,10)==0
            fprintf('.');
        end
        
    end
    acc = correct/total;
fprintf('%d: %f\n',epochs,acc);
LR = LR * 0.98;
end
