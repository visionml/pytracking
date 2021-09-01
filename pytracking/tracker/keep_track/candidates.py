import torch


class Candidate(object):
    def __init__(self, id, score, tsm_coord, object_id):
        self.ids = [id]
        self.scores = [score]
        self.tsm_coords = [tsm_coord]
        self.object_id = object_id


class CandidateCollection(object):
    def __init__(self, scores, tsm_coords, candidate_selection_is_certain=True):
        self.candidates = {}
        self.object_id_cntr = 0
        self.flag = 'normal'
        self.candidate_id_of_selected_candidate = 0
        self.object_id_of_selected_candidate = 0
        self.candidate_selection_is_certain = candidate_selection_is_certain

        if candidate_selection_is_certain == False:
            self.object_id_of_selected_candidate = 1
            self.object_id_cntr = 1

        for id, (score, tsm_coord) in enumerate(zip(scores.cpu().numpy(), tsm_coords.cpu().numpy())):
            self.candidates[id] = Candidate(id, score, tsm_coord, self.object_id_cntr)
            self.object_id_cntr += 1

    def update(self, scores, tsm_coords, matches, match_scores):
        matches = matches.view(-1)
        match_scores = match_scores.view(-1)

        self._reassign_candidates_according_to_matching(match_scores, matches, scores, tsm_coords)

        object0_detected = self._check_if_object0_is_detected()
        object0_detected = self._check_if_more_suitable_candidate_is_available(object0_detected)

        if not object0_detected:
            self._clean_up_if_object0_not_found()
            self._try_to_reselect_candidate()

    def _reassign_candidates_according_to_matching(self, match_scores, matches, scores, tsm_coords):
        candidates = {}
        non_matched_candidates = torch.ones(len(self.candidates))
        # reassign peaks according to matches.
        for id, (score, tsm_coord, match, match_score) in enumerate(zip(scores, tsm_coords, matches, match_scores)):
            if match >= 0:
                candidate = self.candidates[match.item()]
                non_matched_candidates[match.item()] = 0

                is_prob_too_low = (match_score < 0.6 or (match_score < 0.85 and score < 0.2))

                if candidate.object_id == self.object_id_of_selected_candidate and is_prob_too_low:
                    # matching assignment probability is too low skip assignment and start with a new peak.
                    candidate = Candidate(id, score, tsm_coord, self.object_id_cntr)
                    self.object_id_cntr += 1
                else:
                    # add candidate meta data to assigned candidate
                    candidate.scores.append(score)
                    candidate.ids.append(id)
                    candidate.tsm_coords.append(tsm_coord)

                candidates[id] = candidate

            else:
                candidates[id] = Candidate(id, score, tsm_coord, self.object_id_cntr)
                self.object_id_cntr += 1
        self.candidates = candidates

    def _check_if_object0_is_detected(self):
        object0_detected = False
        for id, candidate in self.candidates.items():
            if candidate.object_id == self.object_id_of_selected_candidate:
                self.candidate_id_of_selected_candidate = id
                self.flag = 'normal'
                object0_detected = True

                if max(candidate.scores) > 0.75:
                    self.candidate_selection_is_certain = True

        return object0_detected

    def _check_if_more_suitable_candidate_is_available(self, object0_detected):
        if object0_detected and self.candidate_id_of_selected_candidate != 0:
            # there is a candidate that has a higher score than the currently selected one.
            max_score_candidate = self.candidates[0]
            candidate_selected_as_target = self.candidates[self.candidate_id_of_selected_candidate]

            if max(max_score_candidate.scores) > max(candidate_selected_as_target.scores):
                self.flag = 'normal'
                self.candidate_id_of_selected_candidate = 0
                self.object_id_of_selected_candidate = max_score_candidate.object_id
                object0_detected = True

        return object0_detected

    def _clean_up_if_object0_not_found(self):
        self.candidate_id_of_selected_candidate = None
        # object has just disappeared now.
        if self.flag == 'normal':
            self.flag = 'not_found'
            self.candidate_selection_is_certain = False

    def _try_to_reselect_candidate(self):
        max_score = 0
        for id in self.candidates.keys():
            candidate = self.candidates[id]
            recent_score = candidate.scores[-1]

            if (recent_score > 0.25 and recent_score > max_score):
                self.flag = 'normal'
                self.candidate_id_of_selected_candidate = id
                self.object_id_of_selected_candidate = self.candidates[id].object_id
                max_score = recent_score
