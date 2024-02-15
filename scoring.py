import numpy as np
import pandas as pd

def _analyze_tracks(truth, submission):
    """Compute the majority particle, hit counts, and weight for each track.

    Parameters
    ----------
    truth : pandas.DataFrame
        Truth information. Must have hit_id, particle_id, and weight columns.
    submission : pandas.DataFrame
        Proposed hit/track association. Must have hit_id and track_id columns.

    Returns
    -------
    pandas.DataFrame
        Contains track_id, nhits, major_particle_id, major_particle_nhits,
        major_nhits, and major_weight columns.
    """
    # true number of hits for each particle_id
    particles_nhits = truth['particle_id'].value_counts(sort=False)
    total_weight = truth['weight'].sum()
    # combined event with minimal reconstructed and truth information
    event = pd.merge(truth[['hit_id', 'particle_id', 'weight']],
                         submission[['hit_id', 'track_id']],
                         on=['hit_id'], how='left', validate='one_to_one')
    event.drop('hit_id', axis=1, inplace=True)
    event.sort_values(by=['track_id', 'particle_id'], inplace=True)

    # ASSUMPTIONs: 0 <= track_id, 0 <= particle_id

    tracks = []
    # running sum for the reconstructed track we are currently in
    rec_track_id = -1
    rec_nhits = 0
    # running sum for the particle we are currently in (in this track_id)
    cur_particle_id = -1
    cur_nhits = 0
    cur_weight = 0
    # majority particle with most hits up to now (in this track_id)
    maj_particle_id = -1
    maj_nhits = 0
    maj_weight = 0

    for hit in event.itertuples(index=False):
        # we reached the next track so we need to finish the current one
        if (rec_track_id != -1) and (rec_track_id != hit.track_id):
            # could be that the current particle is the majority one
            if maj_nhits < cur_nhits:
                maj_particle_id = cur_particle_id
                maj_nhits = cur_nhits
                maj_weight = cur_weight
            # store values for this track
            tracks.append((rec_track_id, rec_nhits, maj_particle_id,
                particles_nhits[maj_particle_id], maj_nhits,
                maj_weight / total_weight))

        # setup running values for next track (or first)
        if rec_track_id != hit.track_id:
            rec_track_id = hit.track_id
            rec_nhits = 1
            cur_particle_id = hit.particle_id
            cur_nhits = 1
            cur_weight = hit.weight
            maj_particle_id = -1
            maj_nhits = 0
            maj_weights = 0
            continue

        # hit is part of the current reconstructed track
        rec_nhits += 1

        # reached new particle within the same reconstructed track
        if cur_particle_id != hit.particle_id:
            # check if last particle has more hits than the majority one
            # if yes, set the last particle as the new majority particle
            if maj_nhits < cur_nhits:
                maj_particle_id = cur_particle_id
                maj_nhits = cur_nhits
                maj_weight = cur_weight
            # reset runnig values for current particle
            cur_particle_id = hit.particle_id
            cur_nhits = 1
            cur_weight = hit.weight
        # hit belongs to the same particle within the same reconstructed track
        else:
            cur_nhits += 1
            cur_weight += hit.weight

    # last track is not handled inside the loop
    if maj_nhits < cur_nhits:
        maj_particle_id = cur_particle_id
        maj_nhits = cur_nhits
        maj_weight = cur_weight
    # store values for the last track
    tracks.append((rec_track_id, rec_nhits, maj_particle_id,
        particles_nhits[maj_particle_id], maj_nhits, maj_weight / total_weight))

    cols = ['track_id', 'nhits',
            'major_particle_id', 'major_particle_nhits',
            'major_nhits', 'major_weight']
    return pd.DataFrame.from_records(tracks, columns=cols)

def score_event(truth, submission):
    """Compute the TrackML event score for a single event.

    Parameters
    ----------
    truth : pandas.DataFrame
        Truth information. Must have hit_id, particle_id, and weight columns.
    submission : pandas.DataFrame
        Proposed hit/track association. Must have hit_id and track_id columns.
    """
    tracks = _analyze_tracks(truth, submission)
    purity_rec = np.true_divide(tracks['major_nhits'], tracks['nhits'])
    purity_maj = np.true_divide(tracks['major_nhits'], tracks['major_particle_nhits'])
    good_track = (0.5 < purity_rec) & (0.5 < purity_maj)
    return tracks['major_weight'][good_track].sum()


def false_positive_rate(pred_lbl, true_lbl):
    fp_rate = 0
    for i in range(len(true_lbl)): #go over every single hit from the event
        truth_rows, pred_rows = [], []
        for ind, part in enumerate(true_lbl[i]):
            truth_rows.append((ind, part[0].item(), part[1].item()))

        for ind, pred in enumerate(pred_lbl[i]):
            pred_rows.append((ind, pred.item()))

        truth = pd.DataFrame(truth_rows)
        truth.columns = ['hit_id', 'particle_id', 'weight']
        submission = pd.DataFrame(pred_rows)
        submission.columns = ['hit_id', 'track_id']
        
        tracks_info = _analyze_tracks(truth, submission)
        print(tracks_info)

        true_negatives, false_positives = [], []
        for tid, pid in enumerate(tracks_info['major_particle_id']):
            hits_not_tid = submission[submission['track_id'] != tid]['hit_id'].values
            hits_not_pid = truth[truth['particle_id'] != pid]['hit_id'].values
            true_negatives_pid = len([value for value in hits_not_tid if value in hits_not_pid])
            true_negatives.append(true_negatives_pid)

            hits_tid = submission[submission['track_id'] == tid]['hit_id'].values
            false_positives_pid = len([value for value in hits_tid if value in hits_not_pid])
            false_positives.append(false_positives_pid)


        print(false_positives, true_negatives)
        fp_rates = np.divide(false_positives, [sum(x) for x in zip(false_positives, true_negatives)])
        print(fp_rates)
        fp_rate += np.mean(fp_rates)
    return np.mean(fp_rate)

        

def calc_score(pred_lbl, true_lbl):
    """
    pred_lbl is a tensor containing predicted cluster IDs from a single event
    true_lbl is a tensor containing the true cluster IDs from a single event (track_id from dataset)
    """
    score = 0.
    for i in range(len(true_lbl)): #go over every single hit from the event
        truth_rows, pred_rows = [], []
        for ind, part in enumerate(true_lbl[i]):
            truth_rows.append((ind, part.item(), 1))

        for ind, pred in enumerate(pred_lbl[i]):
            pred_rows.append((ind, pred.item()))

        truth = pd.DataFrame(truth_rows)
        truth.columns = ['hit_id', 'particle_id', 'weight']
        submission = pd.DataFrame(pred_rows)
        submission.columns = ['hit_id', 'track_id']
        score += score_event(truth, submission)
    return score/len(true_lbl)

def calc_score_trackml(pred_lbl, true_lbl):
    """
    pred_lbl is a tensor containing predicted cluster IDs from a single event
    true_lbl is a tensor containing the true cluster IDs from a single event (track_id from dataset)
    """
    truth_rows, pred_rows = [], []
    for ind, part in enumerate(true_lbl):
        truth_rows.append((ind, part[0].item(), part[1].item()))

    for ind, pred in enumerate(pred_lbl):
        pred_rows.append((ind, pred.item()))

    truth = pd.DataFrame(truth_rows)
    truth.columns = ['hit_id', 'particle_id', 'weight']
    submission = pd.DataFrame(pred_rows)
    submission.columns = ['hit_id', 'track_id']
    return score_event(truth, submission)