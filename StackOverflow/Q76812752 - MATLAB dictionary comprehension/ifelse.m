% https://stackoverflow.com/q/76812752/10133797
% Thanks @user16372530 https://stackoverflow.com/a/73751467/10133797
function out = ifelse(a, cond, b)
    if cond
        out = a();
    else
        out = b();
    end
end
