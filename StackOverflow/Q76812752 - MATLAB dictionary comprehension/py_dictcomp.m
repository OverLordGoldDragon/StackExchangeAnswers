% https://stackoverflow.com/q/76812752/10133797
function out = py_dictcomp(fn_k, fn_v, iterable, cond, cell_values)
    % py_dictcomp
    % Mimics Python's
    %
    %     {f(k): g(v) for k, v in dict.items() if cond(k, v)}
    %
    % where [f(k), g(v)] = fn(k, v)
    %
    % or, if `iterable` isn't `dictionary`,
    %
    %     {k: v for i, x in enumerate(iterable) if cond(i, x)}
    %
    % where `[k, v] = f(i, x)`.
    %
    % `cell_values=true` makes `out` like `dictionary(a={1})`. By default,
    % `false` is attempted first. Has no effect if `iterable` is `dictionary`. 
    %
    % Default `cond = @(a, b) true`.
    %
    if nargin == 3
        cond = @(a, b) true;
    end
    if nargin <= 4
        cell_values = false;
    end
    user_set_cell_values = (nargin == 5);

    out = dictionary();
    if strcmp(class(iterable), "dictionary")
        cell_values = strcmp(iterable.values, "cell");
        keys = iterable.keys;
        values = iterable.values;

        for l=1:iterable.numEntries
            k = keys{l};
            if cell_values
                v = values{l};
            else
                v = values(l);
            end
            if cond(k, v)
                k = fn_k(k);
                v = fn_v(v);
                if cell_values
                    out{k} = v;
                else
                    out(k) = v;
                end
            end
        end
    else
        is_cell = iscell(iterable);
        for i=1:numel(iterable)
            if is_cell
                x = iterable{i};
            else
                x = iterable(i);
            end
            
            if cond(i, x)
                k = fn_k(i);
                v = fn_v(x);

                if cell_values
                    out{k} = v;
                else
                    try
                        out(k) = v;
                        % e.g. first populate with `double` values, then
                        % `string`, the `string` becomes `NaN` unless cell
                        if isnan(out(k)) && ~isnan(v)
                            error("NaN `out(k)` without NAN `v`")
                        end
                    catch e
                        if user_set_cell_values
                            throw(e)
                        else
                            out = py_dictcomp(fn_k, fn_v, iterable, cond, true);
                        end
                    end
                end
            end
        end
    end
end
