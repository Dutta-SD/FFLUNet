import torch
import torch.nn as nn
from torch.nn.functional import softmax, one_hot


def get_one_hot(y_true, n_classes):
    y_true = y_true.to(torch.int64)
    y_true = one_hot(y_true, num_classes=n_classes)
    y_true = torch.transpose(y_true, dim0=5, dim1=1)
    y_true = torch.squeeze(y_true, dim=5)
    y_true = y_true.to(torch.int8)
    return y_true


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1e-6
        self.axes = (2, 3, 4)

    def forward(self, y_true, y_pred):
        # Prepare inputs
        y_true = get_one_hot(y_true, y_pred.shape[1])
        y_pred = softmax(y_pred, dim=1)

        num = torch.sum(torch.square(y_true - y_pred), dim=self.axes)
        den = (
            torch.sum(torch.square(y_true), dim=self.axes)
            + torch.sum(torch.square(y_pred), dim=self.axes)
            + self.smooth
        )

        loss = torch.mean(num / den, axis=1)
        loss = torch.mean(loss)
        return loss


class DiceCELoss(nn.Module):
    def __init__(self):
        super(DiceCELoss, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, y_true, y_pred):
        # Dice loss
        loss_dice = self.dice_loss(y_true, y_pred)

        # Prepare inputs
        y_true = get_one_hot(y_true, y_pred.shape[1]).to(torch.float32)

        # Cross entropy loss
        loss_ce = self.cross_entropy(y_pred, y_true)

        return loss_ce + loss_dice


class WeightedDiceLoss(nn.Module):
    def __init__(self, class_weights):
        super(WeightedDiceLoss, self).__init__()

        # Define class weight scheme
        # Move weights to cuda if already given by user
        if not (class_weights is None):
            self.class_weights = torch.Tensor(class_weights).to("cuda")
        else:
            self.class_weights = None

        self.smooth = 1e-6
        self.axes = (2, 3, 4)

    def forward(self, y_true, y_pred):
        # Prepare inputs
        y_true = get_one_hot(y_true, y_pred.shape[1])
        y_pred = softmax(y_pred, dim=1)

        if self.class_weights is None:
            class_weights = torch.sum(y_true, dim=self.axes)
            class_weights = 1.0 / (torch.square(class_weights) + 1.0)
        else:
            class_weights = self.class_weights

        num = torch.sum(torch.square(y_true - y_pred), dim=self.axes)
        num *= class_weights

        den = torch.sum(torch.square(y_true), dim=self.axes) + torch.sum(
            torch.square(y_pred), dim=self.axes
        )
        den *= class_weights
        den += self.smooth

        loss = torch.sum(num, axis=1) / torch.sum(den, axis=1)
        loss = torch.mean(loss)

        return loss


class WeightedDiceCELoss(nn.Module):
    def __init__(self, class_weights):
        super(WeightedDiceCELoss, self).__init__()

        # Define class weight scheme
        # Move weights to cuda if already given by user
        if not (class_weights is None):
            self.class_weights = torch.Tensor(class_weights).to("cuda")
        else:
            self.class_weights = None

        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        self.weighted_dice_loss = WeightedDiceLoss(class_weights=self.class_weights)

    def forward(self, y_true, y_pred):
        # Dice loss
        loss_weighted_dice = self.weighted_dice_loss(y_true, y_pred)

        # Prepare inputs
        y_true = get_one_hot(y_true, y_pred.shape[1]).to(torch.float32)

        # Weighted cross entropy loss
        loss_weighted_ce = self.cross_entropy(y_pred, y_true)

        return loss_weighted_dice + loss_weighted_ce


class KLDivLoss(nn.Module):
    def forward(self, z_mean, z_log_var):
        loss = -0.5 * (1.0 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
        return torch.mean(loss)


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss()
        self.kl_loss = KLDivLoss()

    def forward(self, y_true, y_pred):
        return self.reconstruction_loss(y_true, y_pred[0]) + self.kl_loss(
            y_pred[1], y_pred[2]
        )


class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()
        self.region_loss = DiceCELoss()

    def forward(self, y_true, y_pred, dtm, alpha):
        # Compute region based loss
        region_loss = self.region_loss(y_true, y_pred)

        # Prepare inputs
        y_pred = softmax(y_pred, dim=1)

        # Compute boundary loss
        boundary_loss = torch.mean(dtm * y_pred)

        return alpha * region_loss + (1.0 - alpha) * boundary_loss


class HDOneSidedLoss(nn.Module):
    def __init__(self):
        super(HDOneSidedLoss, self).__init__()
        self.region_loss = DiceCELoss()

    def forward(self, y_true, y_pred, dtm, alpha):
        # Compute region based loss
        region_loss = self.region_loss(y_true, y_pred)

        # Prepare inputs
        y_true = get_one_hot(y_true, y_pred.shape[1])
        y_pred = softmax(y_pred, dim=1)

        # Compute boundary loss
        boundary_loss = torch.mean(torch.square(y_true - y_pred) * torch.square(dtm))

        return alpha * region_loss + (1.0 - alpha) * boundary_loss


"""
Notes:
- alpha is dynamic as we need to vary it during training
- dtm is the distance transform map of the ground truth
"""


class GenSurfLoss(nn.Module):
    def __init__(self, class_weights):
        super(GenSurfLoss, self).__init__()
        self.region_loss = DiceCELoss()

        # Define class weight scheme
        # Move weights to cuda if already given by user
        if not (class_weights is None):
            self.class_weights = torch.Tensor(class_weights).to("cuda")
        else:
            self.class_weights = None

        self.smooth = 1e-6
        self.axes = (2, 3, 4)

    def forward(self, y_true, y_pred, dtm, alpha):
        # Compute region based loss
        region_loss = self.region_loss(y_true, y_pred)

        # Prepare inputs
        y_true = get_one_hot(y_true, y_pred.shape[1])
        y_pred = softmax(y_pred, dim=1)

        if self.class_weights is None:
            class_weights = torch.sum(y_true, dim=self.axes)
            class_weights = 1.0 / (torch.square(class_weights) + 1.0)
        else:
            class_weights = self.class_weights

        # Compute boundary loss
        # Flip each one-hot encoded class
        y_worst = torch.square(1.0 - y_true)

        num = torch.sum(torch.square(dtm * (y_worst - y_pred)), axis=self.axes)
        num *= class_weights

        den = torch.sum(torch.square(dtm * (y_worst - y_true)), axis=self.axes)
        den *= class_weights
        den += self.smooth

        boundary_loss = torch.sum(num, axis=1) / torch.sum(den, axis=1)
        boundary_loss = torch.mean(boundary_loss)
        boundary_loss = 1.0 - boundary_loss

        return alpha * region_loss + (1.0 - alpha) * boundary_loss


def get_loss(args, **kwargs):
    if args.loss == "dice":
        return DiceLoss()
    elif args.loss == "dice_ce":
        return DiceCELoss()
    elif args.loss == "gdl":
        return WeightedDiceLoss(class_weights=kwargs["class_weights"])
    elif args.loss == "gdl_ce":
        return WeightedDiceCELoss(class_weights=kwargs["class_weights"])
    elif args.loss == "bl":
        return BoundaryLoss()
    elif args.loss == "hl":
        return HDOneSidedLoss()
    elif args.loss == "gsl":
        return GenSurfLoss(class_weights=kwargs["class_weights"])
    else:
        raise ValueError("Invalid loss function")


# Since this does not have any dependency from this package, we can run this as a script

if __name__ == "__main__":
    N_CLASSES = 2
    # Test the loss function
    # Batched
    y_true = torch.randint(0, N_CLASSES, (1, 1, 32, 32, 32))
    y_pred = torch.rand((1, N_CLASSES, 32, 32, 32))
    print(y_pred.shape)
    print(y_true.shape)
    loss = WeightedDiceCELoss(None)
    print(loss(y_true, y_pred))
