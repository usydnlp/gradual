import torch
import torch.nn as nn


class ADAPT(nn.Module):

    def __init__(
        self, k=None, q1_size=None, q2_size=None, v1_size=None, v2_size=None,
        nonlinear_proj=False, groups=1, sg_dim=None,
    ):
        '''
            value_size (int): size of the features from the value matrix
            query_size (int): size of the global query vector
            k (int, optional): only used for non-linear projection
            nonlinear_proj (bool): whether to project gamma and beta non-linearly
            groups (int): number of feature groups (default=1)
        '''
        super().__init__()

        #self.query_size = query_size
        self.groups = groups

        
        if nonlinear_proj:
            self.fc_gamma = nn.Sequential(
                nn.Linear(query_size, value_size//k),
                nn.ReLU(inplace=True),
                nn.Linear(value_size//k, value_size),
            )

            self.fc_beta = nn.Sequential(
                nn.Linear(query_size, value_size//k),
                nn.ReLU(inplace=True),
                nn.Linear(value_size//k, value_size),
            )
        else:
            print("Initializing linear ADAPT")

            # Q1 adapter
            if q1_size != v1_size:
                self.v1_transform= nn.Sequential(
                    nn.Linear(v1_size, q1_size),
                )
                v1_size=q1_size
                
            self.fc_gamma = nn.Sequential(
                nn.Linear(q1_size, v1_size//groups),
            )

            self.fc_beta = nn.Sequential(
                nn.Linear(q1_size, v1_size//groups),
            )

            # V2 adapter
            if v2_size is not None:
                
                self.imgsg_beta = nn.Sequential(
                    nn.Linear(v2_size, v1_size//groups),
                )

            # Q2 adapter
            if q2_size is not None:
                
                self.txtsg_beta = nn.Sequential(
                    nn.Linear(q2_size, v1_size // groups),
                )


    def forward(self, value1, value2, query1, query2):
        '''

        Adapt embedding matrix (value) given a query vector.
        Dimension order is the same of the convolutional layers.

        Arguments:
            feat_matrix {torch.FloatTensor}
                -- shape: batch, features, timesteps
            cond_vector {torch.FloatTensor}
                -- shape: ([1 or batch], features)

        Returns:
            torch.FloatTensor
                -- shape: batch, features, timesteps

        Special cases:
            When query shape is (1, features) it is performed
            one-to-many embedding adaptation. A single vector is
            used to filter all instances from the value matrix
            leveraging the brodacast of the query vector.
            This is the default option for retrieval.

            When query shape is (batch, features) it is performed
            pairwise embedding adaptation. i.e., adaptation is performed
            line by line, and value and query must be aligned.
            This could be used for VQA or other tasks that don't require
            ranking all instances from a set.

        '''

        #value 1 (img)
        B, D, rk = value1.shape
        Bv, Dv = query1.shape


        if D != Dv:
            value1=value1.permute(0,2,1)
            value1=self.v1_transform(value1).permute(0,2,1) #B, Dv, K

        value1 = value1.view(
            B, Dv//self.groups, self.groups, -1
        )

        # value 2 (imgsg)
        if value2 is not None:
            v2_betas = self.imgsg_beta(value2).view(
                B, Dv//self.groups, 1, 1
            )
        # query1 (caption)
        gammas = self.fc_gamma(query1).view(
            Bv, Dv//self.groups, 1, 1
        )
        betas  = self.fc_beta(query1).view(
            Bv, Dv//self.groups, 1, 1
        )

        # query2 (txtsg)
        if query2 is not None:
            q2_betas = self.txtsg_beta(query2).view(
                Bv, Dv//self.groups, 1, 1
            )


        if query2 is not None and value2 is None: #sg_type: txt or bi_concat
            normalized = value1 * (gammas + 1) + betas + q2_betas
        elif query2 is None and value2 is not None: #sg_type: img
            normalized = value1 * (gammas + 1) + betas + v2_betas
        else: # sg_type: bi_adapt
            normalized = value1 * (gammas + 1) + betas + q2_betas + v2_betas

        normalized = normalized.view(B, Dv, -1)
        return normalized
