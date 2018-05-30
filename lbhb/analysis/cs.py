def _days_re_exposure(row):
    td = row['date'] - row['exposure_date']
    days = td.days
    if days < 0:
        measure = 'baseline'
    elif days == 1:
        measure = 'd1'
    elif days >= 13:
        measure = 'w2'
    else:
        measure = None
    row['days'] = days
    row['measure'] = measure
    return row


def add_days_re_exposure(df):
    return df.apply(_days_re_exposure, axis=1)

def mark_replicates(df):

    def _helper(row):
        key = row['animal'], row['measure']
        date = row['date']
        replicate_map = replicate_maps[key]
        return replicate_map[date]

    replicates = df.groupby(['animal', 'measure'])['date'].unique()
    replicate_maps = replicates.apply(lambda x: {d: i for i, d in enumerate(x)})
    df['replicate'] = df.apply(_helper, axis=1)
    return df
