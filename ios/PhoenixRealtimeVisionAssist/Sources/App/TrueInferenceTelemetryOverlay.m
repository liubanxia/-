#import <UIKit/UIKit.h>
#import <notify.h>

static NSString * const LiteViewTelemetryNotificationName = @"com.phoenix.realtimevisionassist.broadcast.true-inference.v1";
static NSInteger const LiteViewTelemetryOverlayTag = 84015;
static int LiteViewTelemetryToken = -1;

@interface LiteViewTrueInferenceTelemetryOverlay : NSObject
@end

@implementation LiteViewTrueInferenceTelemetryOverlay

+ (void)load {
    dispatch_async(dispatch_get_main_queue(), ^{
        notify_register_check(LiteViewTelemetryNotificationName.UTF8String, &LiteViewTelemetryToken);
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(sceneDidActivate:)
                                                     name:UISceneDidActivateNotification
                                                   object:nil];
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(sceneDidActivate:)
                                                     name:UIApplicationDidBecomeActiveNotification
                                                   object:nil];
        [NSTimer scheduledTimerWithTimeInterval:0.5
                                         target:self
                                       selector:@selector(refreshTimer:)
                                       userInfo:nil
                                        repeats:YES];
        [self installOrRefresh];
    });
}

+ (void)sceneDidActivate:(NSNotification *)notification {
    [self installOrRefresh];
}

+ (void)refreshTimer:(NSTimer *)timer {
    if (UIApplication.sharedApplication.applicationState == UIApplicationStateActive) {
        [self installOrRefresh];
    }
}

+ (UIWindow *)activeWindow {
    for (UIScene *scene in UIApplication.sharedApplication.connectedScenes) {
        if (![scene isKindOfClass:UIWindowScene.class]) { continue; }
        UIWindowScene *windowScene = (UIWindowScene *)scene;
        if (windowScene.activationState != UISceneActivationStateForegroundActive &&
            windowScene.activationState != UISceneActivationStateForegroundInactive) {
            continue;
        }
        for (UIWindow *window in windowScene.windows) {
            if (window.isKeyWindow) { return window; }
        }
        if (windowScene.windows.firstObject) { return windowScene.windows.firstObject; }
    }
    return nil;
}

+ (void)installOrRefresh {
    UIWindow *window = [self activeWindow];
    if (!window) { return; }

    UIView *panel = [window viewWithTag:LiteViewTelemetryOverlayTag];
    UILabel *label = nil;
    if (!panel) {
        panel = [[UIView alloc] initWithFrame:CGRectZero];
        panel.tag = LiteViewTelemetryOverlayTag;
        panel.translatesAutoresizingMaskIntoConstraints = NO;
        panel.backgroundColor = [UIColor colorWithWhite:0.04 alpha:0.88];
        panel.layer.cornerRadius = 14.0;
        panel.layer.masksToBounds = YES;
        panel.userInteractionEnabled = NO;

        label = [[UILabel alloc] initWithFrame:CGRectZero];
        label.tag = LiteViewTelemetryOverlayTag + 1;
        label.translatesAutoresizingMaskIntoConstraints = NO;
        label.numberOfLines = 0;
        label.font = [UIFont monospacedSystemFontOfSize:11.5 weight:UIFontWeightSemibold];
        label.textColor = UIColor.whiteColor;
        label.adjustsFontSizeToFitWidth = YES;
        label.minimumScaleFactor = 0.78;
        [panel addSubview:label];
        [window addSubview:panel];

        UILayoutGuide *safe = window.safeAreaLayoutGuide;
        [NSLayoutConstraint activateConstraints:@[
            [panel.leadingAnchor constraintEqualToAnchor:safe.leadingAnchor constant:12.0],
            [panel.trailingAnchor constraintEqualToAnchor:safe.trailingAnchor constant:-12.0],
            [panel.bottomAnchor constraintEqualToAnchor:safe.bottomAnchor constant:-10.0],
            [label.leadingAnchor constraintEqualToAnchor:panel.leadingAnchor constant:12.0],
            [label.trailingAnchor constraintEqualToAnchor:panel.trailingAnchor constant:-12.0],
            [label.topAnchor constraintEqualToAnchor:panel.topAnchor constant:9.0],
            [label.bottomAnchor constraintEqualToAnchor:panel.bottomAnchor constant:-9.0]
        ]];
    } else {
        label = (UILabel *)[panel viewWithTag:LiteViewTelemetryOverlayTag + 1];
    }

    if (!label) { return; }
    label.text = [self telemetryText];
    [window bringSubviewToFront:panel];
}

+ (NSString *)telemetryText {
    if (LiteViewTelemetryToken < 0) {
        return @"真推理遥测 0.8.4 · 通道不可用";
    }

    uint64_t state = 0;
    if (notify_get_state(LiteViewTelemetryToken, &state) != NOTIFY_STATUS_OK ||
        (state & (UINT64_C(1) << 63)) == 0) {
        return @"真推理遥测 0.8.4 · 等待 ReplayKit 首次分析";
    }

    uint64_t coreML = state & UINT64_C(0x0FFF);
    uint64_t decoded = (state >> 12) & UINT64_C(0x0FFF);
    uint64_t nonEmpty = (state >> 24) & UINT64_C(0x0FFF);
    uint64_t failovers = (state >> 36) & UINT64_C(0x003F);
    uint64_t failures = (state >> 42) & UINT64_C(0x003F);
    uint64_t model = (state >> 48) & UINT64_C(0x0003);
    uint64_t decoder = (state >> 50) & UINT64_C(0x0007);
    uint64_t source = (state >> 53) & UINT64_C(0x0003);
    uint64_t sequence = (state >> 55) & UINT64_C(0x00FF);

    return [NSString stringWithFormat:
            @"真推理遥测 0.8.4 / Build 15 · hot 384px\n"
             "Core ML %llu · 解码 %llu · 非空目标 %llu · 切备用 %llu · 失败 %llu\n"
             "模型 %@ · 解码路径 %@ · 当前来源 %@ · seq %llu",
            coreML,
            decoded,
            nonEmpty,
            failovers,
            failures,
            [self modelName:model],
            [self decoderName:decoder],
            [self sourceName:source],
            sequence];
}

+ (NSString *)modelName:(uint64_t)code {
    switch (code) {
        case 1: return @"yolo11n";
        case 2: return @"YOLOv3TinyInt8LUT";
        case 3: return @"其他 Core ML";
        default: return @"尚未加载";
    }
}

+ (NSString *)decoderName:(uint64_t)code {
    switch (code) {
        case 1: return @"RecognizedObject";
        case 2: return @"coordinates+confidence";
        case 3: return @"Ultralytics raw";
        case 4: return @"empty output";
        case 5: return @"unsupported";
        default: return @"未确定";
    }
}

+ (NSString *)sourceName:(uint64_t)code {
    switch (code) {
        case 1: return @"Core ML 主模型";
        case 2: return @"Vision Tracker";
        case 3: return @"Vision 独立验证";
        default: return @"无目标/待确认";
    }
}

@end
