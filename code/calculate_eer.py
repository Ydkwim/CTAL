#encoding=utf-8
import os
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

# The codes are forked from GE2E's implementation: https://github.com/HarryVolek/PyTorch_Speaker_Verification/blob/master/train_speech_embedder.py#L92

def get_centroids(embeddings):
    centroids = embeddings.mean(dim=1)
    return centroids

def get_utterance_centroids(embeddings):
    """
    Returns the centroids for each utterance of a speaker, where
    the utterance centroid is the speaker centroid without considering
    this utterance

    Shape of embeddings should be:
        (speaker_ct, utterance_per_speaker_ct, embedding_size)
    """
    sum_centroids = embeddings.sum(dim=1)
    # we want to subtract out each utterance, prior to calculating the
    # the utterance centroid
    sum_centroids = sum_centroids.reshape(
        sum_centroids.shape[0], 1, sum_centroids.shape[-1]
    )
    # we want the mean but not including the utterance itself, so -1
    num_utterances = embeddings.shape[1] - 1
    centroids = (sum_centroids - embeddings) / num_utterances
    return centroids

def get_cossim(embeddings, centroids):
    """
    embeddings 
    centroids: [speaker_ct, embedding_size]
    """
    # number of utterances per speaker
    num_utterances = embeddings.shape[1]
    utterance_centroids = get_utterance_centroids(embeddings)

    # flatten the embeddings and utterance centroids to just utterance,
    # so we can do cosine similarity
    utterance_centroids_flat = utterance_centroids.view(
        utterance_centroids.shape[0] * utterance_centroids.shape[1],
        -1
    ) # [speaker_ct*utterance_per_speaker_ct, embedding_size]
    embeddings_flat = embeddings.reshape(
        embeddings.shape[0] * num_utterances,
        -1
    ) # [speaker_ct*utterance_per_speaker_ct, embedding_size]

    # the cosine distance between utterance and the associated centroids
    # for that utterance
    # this is each speaker's utterances against his own centroid, but each
    # comparison centroid has the current utterance removed
    cos_same = F.cosine_similarity(embeddings_flat, utterance_centroids_flat)

    # now we get the cosine distance between each utterance and the other speakers'
    # centroids
    # to do so requires comparing each utterance to each centroid. To keep the
    # operation fast, we vectorize by using matrices L (embeddings) and
    # R (centroids) where L has each utterance repeated sequentially for all
    # comparisons and R has the entire centroids frame repeated for each utterance
    centroids_expand = centroids.repeat((num_utterances * embeddings.shape[0], 1)) # [centroid_ct*speaker_ct*utterance_per_speaker_ct, embedding_size]
    embeddings_expand = embeddings_flat.unsqueeze(1).repeat(1, embeddings.shape[0], 1) # [speaker_ct*utterance_per_speaker_ct, speaker_ct, embedding_size]
    embeddings_expand = embeddings_expand.view(
        embeddings_expand.shape[0] * embeddings_expand.shape[1],
        embeddings_expand.shape[-1]
    ) # [speaker_ct*utterance_per_speaker_ct*speaker_ct, embedding_size]
    cos_diff = F.cosine_similarity(embeddings_expand, centroids_expand)
    cos_diff = cos_diff.view(
        embeddings.size(0),
        num_utterances,
        centroids.size(0)
    ) # [spekaer_ct, utterance_per_speaker_ct, centroid_ct]
    # assign the cosine distance for same speakers to the proper idx
    same_idx = list(range(embeddings.size(0)))
    cos_diff[same_idx, :, same_idx] = cos_same.view(embeddings.shape[0], num_utterances)
    cos_diff = cos_diff + 1e-6
    return cos_diff


def get_eer(preds, targets, debug=False):

    speaker2embeddings = {}

    for i in range(len(targets)):
        sp = targets[i]
        embedding = preds[i]

        if sp not in speaker2embeddings:
            speaker2embeddings[sp] = []

        speaker2embeddings[sp].append(embedding)

    for sp in speaker2embeddings:
        speaker2embeddings[sp] = np.stack(speaker2embeddings[sp], axis=0)

    N = 4
    M = 50

    avg_EER = 0
    for e in tqdm(range(10)):

        batch_avg_EER = 0
        for batch_id, _ in enumerate(speaker2embeddings):

            speakers = random.sample(speaker2embeddings.keys(), N)

            all_utterances = []

            for speaker in speakers:
                speaker_npy = np.array(speaker2embeddings[speaker])
                utter_index = np.random.randint(0, speaker_npy.shape[0], M)

                utterance = speaker_npy[utter_index]  # [M, hidden_dim]

                all_utterances.append(utterance)

            all_utterances = np.stack(all_utterances, axis=0)  # [N, M, hidden_dim]
            all_utterances = torch.from_numpy(all_utterances)

            enrollment_embeddings, verification_embeddings = torch.split(all_utterances, int(M / 2), dim=1)
            enrollment_centroids = get_centroids(enrollment_embeddings)

            sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)

            # calculating EER
            diff = 1;
            EER = 0;
            EER_thresh = 0;
            EER_FAR = 0;
            EER_FRR = 0

            for thres in [0.01 * i for i in range(101)]:
                sim_matrix_thresh = sim_matrix > thres

                FAR = (sum([sim_matrix_thresh[i].float().sum() - sim_matrix_thresh[i, :, i].float().sum() for i in
                            range(int(N))])
                       / (N - 1.0) / (float(M / 2)) / N)

                FRR = (sum([M / 2 - sim_matrix_thresh[i, :, i].float().sum() for i in range(int(N))])
                       / (float(M / 2)) / N)

                # Save threshold when FAR = FRR (=EER)
                if diff > abs(FAR - FRR):
                    diff = abs(FAR - FRR)
                    EER = (FAR + FRR) / 2
                    EER_thresh = thres
                    EER_FAR = FAR
                    EER_FRR = FRR
            batch_avg_EER += EER
            if debug:
                print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)"%(EER,EER_thresh,EER_FAR,EER_FRR))
        avg_EER += batch_avg_EER / (batch_id + 1)
    avg_EER = avg_EER / 10
    return avg_EER


