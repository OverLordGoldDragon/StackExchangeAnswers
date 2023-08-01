% https://stackoverflow.com/q/76812752/10133797
%% {"a": 5 for i, x in enumerate([1, 2, 3])}
out = py_dictcomp(@(i)i, @(v)5, [1 2 3]);

ref = dictionary();
for i=1:3
    ref(i) = 5;
end
assert(isequal(out, ref))

%% {2*i: x-1 for i, x in enumerate([1, 2, 3]) if (x != 2 and i > 0)}
out = py_dictcomp(@(i)2*i, @(x)x - 1, [1 2 3], @(i, x)x ~= 2 && i > 1);

ref = dictionary();
ref(6) = 2;
assert(isequal(out, ref))

%% {i: ("a", x**2) for i, x in enumerate([1, np.array([2, 3])])}
out = py_dictcomp(@(i)i, @(x){"a", x.^2}, {1, [2 3]});

ref = dictionary();
cl = {1, [2 3]};
for i=1:2
    ref{i} = {"a", cl{i}.^2};
end
assert(isequal(out, ref))

%% {k: v**2 for k, v in dict(a=1, b=-2).items()}
out = py_dictcomp(@(k)k, @(v)v^2, dictionary(a=1, b=-2));
ref = dictionary(a=1^2, b=(-2)^2);
assert(isequal(out, ref))

%% {k: v**2 for k, v in dict(a=1, b=-2).items() if v > 0}
out = py_dictcomp(@(k)k, @(v)v^2, dictionary(a=1, b=-2), @(k, v)v > 0);
ref = dictionary(a=1^2);
assert(isequal(out, ref))

%% {("we have" if i == 0 else i - 1): 
%   ((x + "ogs") if isinstance(x, str) else (x + 1))
%   for i, x in enumerate([2, "d", 3.5])
%   if not isinstance(x, float)}
out = py_dictcomp(@(i)ifelse("we have", i == 1, i - 1), ...
                  @(x)ifelse(x + "ogs", isstring(x), x + 1), ...
                  {2, "d", 3.5}, ...
                  @(i, x)~(isnumeric(x) && mod(x, 1) ~= 0));

%% invalid `cell_values=false`
try
    out = py_dictcomp(@(i)i, @(x)x, {1, [2, 3]}, @(a, b)true, false);
    assert(false);
catch e
    assert(contains(e.identifier, "KeyValueDimsMustMatch"))
end
