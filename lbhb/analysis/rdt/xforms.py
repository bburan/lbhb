def format_keywordstring(recording, keywordstring, **context):
    n_targets = recording.meta['n_targets']
    return {
        'keywordstring': keywordstring.format(n_targets=n_targets),
    }


def fix_modelspec(modelspec, **context):
    for ms in modelspec.raw:
        ms[-1]['fn_kwargs']['i'] = 'pred'
    return {'modelspec': modelspec}
